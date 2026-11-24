#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — CLIP — عندما تتعلّم الصورة والكلمة لغة مشتركة
Large Language Model Engineering — CLIP

Book slug : llm-engineering-69-chapters
Chapter   : ch-40-chapter-40 (lesson 40)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-40-chapter-40.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — CLIP — عندما تتعلّم الصورة والكلمة لغة مشتركة
Large Language Model Engineering — CLIP

Book slug : llm-engineering-69-chapters
Chapter   : ch-40-chapter-40 (lesson 40)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-40-chapter-40.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — CLIP — عندما تتعلّم الصورة والكلمة لغة مشتركة
Large Language Model Engineering — CLIP

Book slug : llm-engineering-69-chapters
Chapter   : ch-40-chapter-40 (lesson 40)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_40.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 40: الفصل 40: CLIP — عندما تتعلّم الصورة والكلمة لغة مشتركة."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def cosine(left: list[float], right: list[float]) -> float:
    denominator = math.sqrt(sum((x * x for x in left)) * sum((y * y for y in right)))
    return sum((x * y for x, y in zip(left, right))) / denominator if denominator else 0.0

def run_clip() -> None:
    images = [[1.0, 0.0], [0.0, 1.0]]
    texts = [[0.9, 0.1], [0.1, 0.9]]
    scores = [[cosine(image, text) / 0.07 for text in texts] for image in images]
    print('image-text similarity logits:')
    for row in scores:
        print(' ', [round(x, 2) for x in row])
    print('matched-pair retrieval:', [max(range(2), key=row.__getitem__) for row in scores])

if __name__ == "__main__":
    print("Chapter 40 lab: Rank images with a shared-space CLIP similarity")
    run_clip()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-08-multimodal-models/chapter-40.md
# ────────────────────────────────────────────────────────────────────────
"""
CLIP مصغّر من الصفر — تدريب تبايني (Contrastive) بين صور ونصوص
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

# أسماء فئات Fashion-MNIST — ستكون "المفردات" النصية لدينا
CLASS_NAMES = [
    "تي شيرت", "بنطال", "بلوفر", "فستان", "معطف",
    "صندل", "قميص", "حذاء رياضي", "حقيبة", "حذاء كاحل",
]

# قاموس حروف بسيط لترميز النصوص العربية حرفيًا (توضيحي وبسيط للتجربة)
VOCAB = sorted(set("".join(CLASS_NAMES) + " صورةهذهمنفئة"))
CHAR_TO_ID = {ch: i + 1 for i, ch in enumerate(VOCAB)}  # 0 محجوز للـ padding
VOCAB_SIZE = len(VOCAB) + 1
MAX_TEXT_LEN = 24


def encode_text(text: str) -> torch.Tensor:
    ids = [CHAR_TO_ID.get(ch, 0) for ch in text[:MAX_TEXT_LEN]]
    ids += [0] * (MAX_TEXT_LEN - len(ids))
    return torch.tensor(ids, dtype=torch.long)


# ---------------------------------------------------------------------
# 1) مجموعة بيانات (صورة، نص) مبنية من Fashion-MNIST
# ---------------------------------------------------------------------
class ImageTextDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        caption = f"صورة من فئة {CLASS_NAMES[label]}"
        text_ids = encode_text(caption)
        return image, text_ids, label


# ---------------------------------------------------------------------
# 2) مشفّر الصور (Image Encoder) — CNN بسيط (بديل مبسّط عن ViT)
# ---------------------------------------------------------------------
class ImageEncoder(nn.Module):
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 28 -> 14
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 14 -> 7
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Linear(128, embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.conv_layers(images).flatten(1)
        return self.projection(features)


# ---------------------------------------------------------------------
# 3) مشفّر النصوص (Text Encoder) — Transformer نصي بسيط
# ---------------------------------------------------------------------
class TextEncoder(nn.Module):
    def __init__(
        self, vocab_size: int, embed_dim: int = 128, num_heads: int = 4, depth: int = 2
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, MAX_TEXT_LEN, embed_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, text_ids: torch.Tensor) -> torch.Tensor:
        padding_mask = text_ids == 0
        x = self.token_embedding(text_ids) + self.position_embedding
        x = self.encoder(x, src_key_padding_mask=padding_mask)

        # استخلاص تمثيل أول رمز غير محشو كـ "ملخّص" للجملة (مشابه لـ [EOS] pooling)
        lengths = (~padding_mask).sum(dim=1) - 1
        lengths = lengths.clamp(min=0)
        pooled = x[torch.arange(x.size(0)), lengths]
        return self.projection(pooled)


# ---------------------------------------------------------------------
# 4) نموذج CLIP الكامل
# ---------------------------------------------------------------------
class MiniCLIP(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder = TextEncoder(vocab_size, embed_dim)
        # معامل درجة الحرارة القابل للتعلّم — يبدأ بما يوافق 1/0.07 كما في CLIP الأصلي
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        features = self.image_encoder(images)
        return F.normalize(features, dim=-1)

    def encode_text(self, text_ids: torch.Tensor) -> torch.Tensor:
        features = self.text_encoder(text_ids)
        return F.normalize(features, dim=-1)

    def forward(self, images: torch.Tensor, text_ids: torch.Tensor):
        image_features = self.encode_image(images)
        text_features = self.encode_text(text_ids)

        # تقييد logit_scale لتفادي الانفجار العددي (كما في CLIP الأصلي، يُقيَّد بـ 100)
        logit_scale = self.logit_scale.exp().clamp(max=100.0)

        # مصفوفة التشابه N x N
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text


# ---------------------------------------------------------------------
# 5) دالة خسارة CLIP التبايني المتناظر (Symmetric InfoNCE)
# ---------------------------------------------------------------------
def clip_loss(logits_per_image: torch.Tensor, logits_per_text: torch.Tensor) -> torch.Tensor:
    batch_size = logits_per_image.size(0)
    # العلامات الصحيحة هي ببساطة القطر: العنصر i يطابق العنصر i
    ground_truth = torch.arange(batch_size, device=logits_per_image.device)

    loss_image_to_text = F.cross_entropy(logits_per_image, ground_truth)
    loss_text_to_image = F.cross_entropy(logits_per_text, ground_truth)

    return (loss_image_to_text + loss_text_to_image) / 2.0


# ---------------------------------------------------------------------
# 6) حلقة التدريب
# ---------------------------------------------------------------------
def train_clip(model: nn.Module, train_loader: DataLoader, num_epochs: int = 10):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, text_ids, _ in train_loader:
            images, text_ids = images.to(device), text_ids.to(device)

            optimizer.zero_grad()
            logits_per_image, logits_per_text = model(images, text_ids)
            loss = clip_loss(logits_per_image, logits_per_text)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # تقييد logit_scale يدويًا بعد كل خطوة (ممارسة قياسية في CLIP)
            with torch.no_grad():
                model.logit_scale.clamp_(max=math.log(100.0))

            running_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch+1:02d}/{num_epochs} | Loss: {running_loss/len(train_loader):.4f}")


# ---------------------------------------------------------------------
# 7) الاستدلال: Zero-Shot Classification
# ---------------------------------------------------------------------
@torch.no_grad()
def zero_shot_classify(model: MiniCLIP, images: torch.Tensor, class_names: list[str]) -> torch.Tensor:
    """
    صنّف صورًا دون أي تدريب إضافي: حوّل كل فئة إلى جملة نصية،
    قارن تمثيل كل صورة بتمثيل كل جملة، واختر الأعلى تشابهًا.
    """
    model.eval()
    prompts = [f"صورة من فئة {name}" for name in class_names]
    text_ids = torch.stack([encode_text(p) for p in prompts]).to(device)

    text_features = model.encode_text(text_ids)        # [num_classes, D]
    image_features = model.encode_image(images.to(device))  # [B, D]

    similarity = image_features @ text_features.t()     # [B, num_classes]
    predictions = similarity.argmax(dim=-1)
    return predictions


# ---------------------------------------------------------------------
# 8) التشغيل الكامل
# ---------------------------------------------------------------------
if __name__ == "__main__":
    transform = T.Compose([T.ToTensor()])

    train_base = torchvision.datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_base = torchvision.datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_dataset = ImageTextDataset(train_base)
    test_dataset = ImageTextDataset(test_base)

    train_loader = DataLoader(
        train_dataset, batch_size=256, shuffle=True, num_workers=2, drop_last=True
    )

    model = MiniCLIP(vocab_size=VOCAB_SIZE, embed_dim=128)
    train_clip(model, train_loader, num_epochs=10)

    # تقييم Zero-Shot على مجموعة الاختبار
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    correct, total = 0, 0
    for images, _, labels in test_loader:
        predictions = zero_shot_classify(model, images, CLASS_NAMES)
        correct += (predictions.cpu() == labels).sum().item()
        total += labels.size(0)

    print(f"\nدقة Zero-Shot Classification: {correct/total:.2%}")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-08-multimodal-models/chapter-40.md
# ────────────────────────────────────────────────────────────────────────
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = Image.open("cat.jpg")
candidate_labels = ["a photo of a cat", "a photo of a dog", "a photo of a car"]

inputs = processor(
    text=candidate_labels, images=image, return_tensors="pt", padding=True
)
outputs = model(**inputs)

# logits_per_image: [1, num_labels] -> التشابه بين الصورة وكل نص مرشح
probs = outputs.logits_per_image.softmax(dim=1)
for label, prob in zip(candidate_labels, probs[0].tolist()):
    print(f"{label}: {prob:.2%}")

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-08-multimodal-models/chapter-40.md
# ────────────────────────────────────────────────────────────────────────
def build_image_index(image_paths: list[str], model, processor) -> torch.Tensor:
    """
    مرّر كل الصور عبر Image Encoder، استخرج تمثيلاتها المطبّعة،
    واحفظها في مصفوفة فهرس واحدة [num_images, embed_dim].
    """
    ...

def search_by_text(query: str, image_index: torch.Tensor, image_paths: list[str],
                    model, processor, top_k: int = 5) -> list[tuple[str, float]]:
    """
    حوّل الاستعلام النصي إلى تمثيل، احسب التشابه مع كل الصور المفهرسة،
    أعد أفضل top_k صورة مع درجة تشابهها.
    """
    ...

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-40-chapter-40.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 40: الفصل 40: CLIP — عندما تتعلّم الصورة والكلمة لغة مشتركة."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def cosine(left: list[float], right: list[float]) -> float:
    denominator = math.sqrt(sum((x * x for x in left)) * sum((y * y for y in right)))
    return sum((x * y for x, y in zip(left, right))) / denominator if denominator else 0.0

def run_clip() -> None:
    images = [[1.0, 0.0], [0.0, 1.0]]
    texts = [[0.9, 0.1], [0.1, 0.9]]
    scores = [[cosine(image, text) / 0.07 for text in texts] for image in images]
    print('image-text similarity logits:')
    for row in scores:
        print(' ', [round(x, 2) for x in row])
    print('matched-pair retrieval:', [max(range(2), key=row.__getitem__) for row in scores])

if __name__ == "__main__":
    print("Chapter 40 lab: Rank images with a shared-space CLIP similarity")
    run_clip()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-40-chapter-40.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""
CLIP مصغّر من الصفر — تدريب تبايني (Contrastive) بين صور ونصوص
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

# أسماء فئات Fashion-MNIST — ستكون "المفردات" النصية لدينا
CLASS_NAMES = [
    "تي شيرت", "بنطال", "بلوفر", "فستان", "معطف",
    "صندل", "قميص", "حذاء رياضي", "حقيبة", "حذاء كاحل",
]

# قاموس حروف بسيط لترميز النصوص العربية حرفيًا (توضيحي وبسيط للتجربة)
VOCAB = sorted(set("".join(CLASS_NAMES) + " صورةهذهمنفئة"))
CHAR_TO_ID = {ch: i + 1 for i, ch in enumerate(VOCAB)}  # 0 محجوز للـ padding
VOCAB_SIZE = len(VOCAB) + 1
MAX_TEXT_LEN = 24


def encode_text(text: str) -> torch.Tensor:
    ids = [CHAR_TO_ID.get(ch, 0) for ch in text[:MAX_TEXT_LEN]]
    ids += [0] * (MAX_TEXT_LEN - len(ids))
    return torch.tensor(ids, dtype=torch.long)


# ---------------------------------------------------------------------
# 1) مجموعة بيانات (صورة، نص) مبنية من Fashion-MNIST
# ---------------------------------------------------------------------
class ImageTextDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        caption = f"صورة من فئة {CLASS_NAMES[label]}"
        text_ids = encode_text(caption)
        return image, text_ids, label


# ---------------------------------------------------------------------
# 2) مشفّر الصور (Image Encoder) — CNN بسيط (بديل مبسّط عن ViT)
# ---------------------------------------------------------------------
class ImageEncoder(nn.Module):
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 28 -> 14
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 14 -> 7
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Linear(128, embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.conv_layers(images).flatten(1)
        return self.projection(features)


# ---------------------------------------------------------------------
# 3) مشفّر النصوص (Text Encoder) — Transformer نصي بسيط
# ---------------------------------------------------------------------
class TextEncoder(nn.Module):
    def __init__(
        self, vocab_size: int, embed_dim: int = 128, num_heads: int = 4, depth: int = 2
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, MAX_TEXT_LEN, embed_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, text_ids: torch.Tensor) -> torch.Tensor:
        padding_mask = text_ids == 0
        x = self.token_embedding(text_ids) + self.position_embedding
        x = self.encoder(x, src_key_padding_mask=padding_mask)

        # استخلاص تمثيل أول رمز غير محشو كـ "ملخّص" للجملة (مشابه لـ [EOS] pooling)
        lengths = (~padding_mask).sum(dim=1) - 1
        lengths = lengths.clamp(min=0)
        pooled = x[torch.arange(x.size(0)), lengths]
        return self.projection(pooled)


# ---------------------------------------------------------------------
# 4) نموذج CLIP الكامل
# ---------------------------------------------------------------------
class MiniCLIP(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder = TextEncoder(vocab_size, embed_dim)
        # معامل درجة الحرارة القابل للتعلّم — يبدأ بما يوافق 1/0.07 كما في CLIP الأصلي
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        features = self.image_encoder(images)
        return F.normalize(features, dim=-1)

    def encode_text(self, text_ids: torch.Tensor) -> torch.Tensor:
        features = self.text_encoder(text_ids)
        return F.normalize(features, dim=-1)

    def forward(self, images: torch.Tensor, text_ids: torch.Tensor):
        image_features = self.encode_image(images)
        text_features = self.encode_text(text_ids)

        # تقييد logit_scale لتفادي الانفجار العددي (كما في CLIP الأصلي، يُقيَّد بـ 100)
        logit_scale = self.logit_scale.exp().clamp(max=100.0)

        # مصفوفة التشابه N x N
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text


# ---------------------------------------------------------------------
# 5) دالة خسارة CLIP التبايني المتناظر (Symmetric InfoNCE)
# ---------------------------------------------------------------------
def clip_loss(logits_per_image: torch.Tensor, logits_per_text: torch.Tensor) -> torch.Tensor:
    batch_size = logits_per_image.size(0)
    # العلامات الصحيحة هي ببساطة القطر: العنصر i يطابق العنصر i
    ground_truth = torch.arange(batch_size, device=logits_per_image.device)

    loss_image_to_text = F.cross_entropy(logits_per_image, ground_truth)
    loss_text_to_image = F.cross_entropy(logits_per_text, ground_truth)

    return (loss_image_to_text + loss_text_to_image) / 2.0


# ---------------------------------------------------------------------
# 6) حلقة التدريب
# ---------------------------------------------------------------------
def train_clip(model: nn.Module, train_loader: DataLoader, num_epochs: int = 10):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, text_ids, _ in train_loader:
            images, text_ids = images.to(device), text_ids.to(device)

            optimizer.zero_grad()
            logits_per_image, logits_per_text = model(images, text_ids)
            loss = clip_loss(logits_per_image, logits_per_text)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # تقييد logit_scale يدويًا بعد كل خطوة (ممارسة قياسية في CLIP)
            with torch.no_grad():
                model.logit_scale.clamp_(max=math.log(100.0))

            running_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch+1:02d}/{num_epochs} | Loss: {running_loss/len(train_loader):.4f}")


# ---------------------------------------------------------------------
# 7) الاستدلال: Zero-Shot Classification
# ---------------------------------------------------------------------
@torch.no_grad()
def zero_shot_classify(model: MiniCLIP, images: torch.Tensor, class_names: list[str]) -> torch.Tensor:
    """
    صنّف صورًا دون أي تدريب إضافي: حوّل كل فئة إلى جملة نصية،
    قارن تمثيل كل صورة بتمثيل كل جملة، واختر الأعلى تشابهًا.
    """
    model.eval()
    prompts = [f"صورة من فئة {name}" for name in class_names]
    text_ids = torch.stack([encode_text(p) for p in prompts]).to(device)

    text_features = model.encode_text(text_ids)        # [num_classes, D]
    image_features = model.encode_image(images.to(device))  # [B, D]

    similarity = image_features @ text_features.t()     # [B, num_classes]
    predictions = similarity.argmax(dim=-1)
    return predictions


# ---------------------------------------------------------------------
# 8) التشغيل الكامل
# ---------------------------------------------------------------------
if __name__ == "__main__":
    transform = T.Compose([T.ToTensor()])

    train_base = torchvision.datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_base = torchvision.datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_dataset = ImageTextDataset(train_base)
    test_dataset = ImageTextDataset(test_base)

    train_loader = DataLoader(
        train_dataset, batch_size=256, shuffle=True, num_workers=2, drop_last=True
    )

    model = MiniCLIP(vocab_size=VOCAB_SIZE, embed_dim=128)
    train_clip(model, train_loader, num_epochs=10)

    # تقييم Zero-Shot على مجموعة الاختبار
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    correct, total = 0, 0
    for images, _, labels in test_loader:
        predictions = zero_shot_classify(model, images, CLASS_NAMES)
        correct += (predictions.cpu() == labels).sum().item()
        total += labels.size(0)

    print(f"\nدقة Zero-Shot Classification: {correct/total:.2%}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-40-chapter-40.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = Image.open("cat.jpg")
candidate_labels = ["a photo of a cat", "a photo of a dog", "a photo of a car"]

inputs = processor(
    text=candidate_labels, images=image, return_tensors="pt", padding=True
)
outputs = model(**inputs)

# logits_per_image: [1, num_labels] -> التشابه بين الصورة وكل نص مرشح
probs = outputs.logits_per_image.softmax(dim=1)
for label, prob in zip(candidate_labels, probs[0].tolist()):
    print(f"{label}: {prob:.2%}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-40-chapter-40.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def build_image_index(image_paths: list[str], model, processor) -> torch.Tensor:
    """
    مرّر كل الصور عبر Image Encoder، استخرج تمثيلاتها المطبّعة،
    واحفظها في مصفوفة فهرس واحدة [num_images, embed_dim].
    """
    ...

def search_by_text(query: str, image_index: torch.Tensor, image_paths: list[str],
                    model, processor, top_k: int = 5) -> list[tuple[str, float]]:
    """
    حوّل الاستعلام النصي إلى تمثيل، احسب التشابه مع كل الصور المفهرسة،
    أعد أفضل top_k صورة مع درجة تشابهها.
    """
    ...

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-40-chapter-40.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — CLIP — عندما تتعلّم الصورة والكلمة لغة مشتركة
Large Language Model Engineering — CLIP

Book slug : llm-engineering-69-chapters
Chapter   : ch-40-chapter-40 (lesson 40)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_40.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 40: الفصل 40: CLIP — عندما تتعلّم الصورة والكلمة لغة مشتركة."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def cosine(left: list[float], right: list[float]) -> float:
    denominator = math.sqrt(sum((x * x for x in left)) * sum((y * y for y in right)))
    return sum((x * y for x, y in zip(left, right))) / denominator if denominator else 0.0

def run_clip() -> None:
    images = [[1.0, 0.0], [0.0, 1.0]]
    texts = [[0.9, 0.1], [0.1, 0.9]]
    scores = [[cosine(image, text) / 0.07 for text in texts] for image in images]
    print('image-text similarity logits:')
    for row in scores:
        print(' ', [round(x, 2) for x in row])
    print('matched-pair retrieval:', [max(range(2), key=row.__getitem__) for row in scores])

if __name__ == "__main__":
    print("Chapter 40 lab: Rank images with a shared-space CLIP similarity")
    run_clip()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-08-multimodal-models/chapter-40.md
# ────────────────────────────────────────────────────────────────────────
"""
CLIP مصغّر من الصفر — تدريب تبايني (Contrastive) بين صور ونصوص
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

# أسماء فئات Fashion-MNIST — ستكون "المفردات" النصية لدينا
CLASS_NAMES = [
    "تي شيرت", "بنطال", "بلوفر", "فستان", "معطف",
    "صندل", "قميص", "حذاء رياضي", "حقيبة", "حذاء كاحل",
]

# قاموس حروف بسيط لترميز النصوص العربية حرفيًا (توضيحي وبسيط للتجربة)
VOCAB = sorted(set("".join(CLASS_NAMES) + " صورةهذهمنفئة"))
CHAR_TO_ID = {ch: i + 1 for i, ch in enumerate(VOCAB)}  # 0 محجوز للـ padding
VOCAB_SIZE = len(VOCAB) + 1
MAX_TEXT_LEN = 24


def encode_text(text: str) -> torch.Tensor:
    ids = [CHAR_TO_ID.get(ch, 0) for ch in text[:MAX_TEXT_LEN]]
    ids += [0] * (MAX_TEXT_LEN - len(ids))
    return torch.tensor(ids, dtype=torch.long)


# ---------------------------------------------------------------------
# 1) مجموعة بيانات (صورة، نص) مبنية من Fashion-MNIST
# ---------------------------------------------------------------------
class ImageTextDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        caption = f"صورة من فئة {CLASS_NAMES[label]}"
        text_ids = encode_text(caption)
        return image, text_ids, label


# ---------------------------------------------------------------------
# 2) مشفّر الصور (Image Encoder) — CNN بسيط (بديل مبسّط عن ViT)
# ---------------------------------------------------------------------
class ImageEncoder(nn.Module):
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 28 -> 14
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 14 -> 7
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Linear(128, embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.conv_layers(images).flatten(1)
        return self.projection(features)


# ---------------------------------------------------------------------
# 3) مشفّر النصوص (Text Encoder) — Transformer نصي بسيط
# ---------------------------------------------------------------------
class TextEncoder(nn.Module):
    def __init__(
        self, vocab_size: int, embed_dim: int = 128, num_heads: int = 4, depth: int = 2
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, MAX_TEXT_LEN, embed_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, text_ids: torch.Tensor) -> torch.Tensor:
        padding_mask = text_ids == 0
        x = self.token_embedding(text_ids) + self.position_embedding
        x = self.encoder(x, src_key_padding_mask=padding_mask)

        # استخلاص تمثيل أول رمز غير محشو كـ "ملخّص" للجملة (مشابه لـ [EOS] pooling)
        lengths = (~padding_mask).sum(dim=1) - 1
        lengths = lengths.clamp(min=0)
        pooled = x[torch.arange(x.size(0)), lengths]
        return self.projection(pooled)


# ---------------------------------------------------------------------
# 4) نموذج CLIP الكامل
# ---------------------------------------------------------------------
class MiniCLIP(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder = TextEncoder(vocab_size, embed_dim)
        # معامل درجة الحرارة القابل للتعلّم — يبدأ بما يوافق 1/0.07 كما في CLIP الأصلي
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        features = self.image_encoder(images)
        return F.normalize(features, dim=-1)

    def encode_text(self, text_ids: torch.Tensor) -> torch.Tensor:
        features = self.text_encoder(text_ids)
        return F.normalize(features, dim=-1)

    def forward(self, images: torch.Tensor, text_ids: torch.Tensor):
        image_features = self.encode_image(images)
        text_features = self.encode_text(text_ids)

        # تقييد logit_scale لتفادي الانفجار العددي (كما في CLIP الأصلي، يُقيَّد بـ 100)
        logit_scale = self.logit_scale.exp().clamp(max=100.0)

        # مصفوفة التشابه N x N
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text


# ---------------------------------------------------------------------
# 5) دالة خسارة CLIP التبايني المتناظر (Symmetric InfoNCE)
# ---------------------------------------------------------------------
def clip_loss(logits_per_image: torch.Tensor, logits_per_text: torch.Tensor) -> torch.Tensor:
    batch_size = logits_per_image.size(0)
    # العلامات الصحيحة هي ببساطة القطر: العنصر i يطابق العنصر i
    ground_truth = torch.arange(batch_size, device=logits_per_image.device)

    loss_image_to_text = F.cross_entropy(logits_per_image, ground_truth)
    loss_text_to_image = F.cross_entropy(logits_per_text, ground_truth)

    return (loss_image_to_text + loss_text_to_image) / 2.0


# ---------------------------------------------------------------------
# 6) حلقة التدريب
# ---------------------------------------------------------------------
def train_clip(model: nn.Module, train_loader: DataLoader, num_epochs: int = 10):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, text_ids, _ in train_loader:
            images, text_ids = images.to(device), text_ids.to(device)

            optimizer.zero_grad()
            logits_per_image, logits_per_text = model(images, text_ids)
            loss = clip_loss(logits_per_image, logits_per_text)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # تقييد logit_scale يدويًا بعد كل خطوة (ممارسة قياسية في CLIP)
            with torch.no_grad():
                model.logit_scale.clamp_(max=math.log(100.0))

            running_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch+1:02d}/{num_epochs} | Loss: {running_loss/len(train_loader):.4f}")


# ---------------------------------------------------------------------
# 7) الاستدلال: Zero-Shot Classification
# ---------------------------------------------------------------------
@torch.no_grad()
def zero_shot_classify(model: MiniCLIP, images: torch.Tensor, class_names: list[str]) -> torch.Tensor:
    """
    صنّف صورًا دون أي تدريب إضافي: حوّل كل فئة إلى جملة نصية،
    قارن تمثيل كل صورة بتمثيل كل جملة، واختر الأعلى تشابهًا.
    """
    model.eval()
    prompts = [f"صورة من فئة {name}" for name in class_names]
    text_ids = torch.stack([encode_text(p) for p in prompts]).to(device)

    text_features = model.encode_text(text_ids)        # [num_classes, D]
    image_features = model.encode_image(images.to(device))  # [B, D]

    similarity = image_features @ text_features.t()     # [B, num_classes]
    predictions = similarity.argmax(dim=-1)
    return predictions


# ---------------------------------------------------------------------
# 8) التشغيل الكامل
# ---------------------------------------------------------------------
if __name__ == "__main__":
    transform = T.Compose([T.ToTensor()])

    train_base = torchvision.datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_base = torchvision.datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_dataset = ImageTextDataset(train_base)
    test_dataset = ImageTextDataset(test_base)

    train_loader = DataLoader(
        train_dataset, batch_size=256, shuffle=True, num_workers=2, drop_last=True
    )

    model = MiniCLIP(vocab_size=VOCAB_SIZE, embed_dim=128)
    train_clip(model, train_loader, num_epochs=10)

    # تقييم Zero-Shot على مجموعة الاختبار
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    correct, total = 0, 0
    for images, _, labels in test_loader:
        predictions = zero_shot_classify(model, images, CLASS_NAMES)
        correct += (predictions.cpu() == labels).sum().item()
        total += labels.size(0)

    print(f"\nدقة Zero-Shot Classification: {correct/total:.2%}")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-08-multimodal-models/chapter-40.md
# ────────────────────────────────────────────────────────────────────────
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = Image.open("cat.jpg")
candidate_labels = ["a photo of a cat", "a photo of a dog", "a photo of a car"]

inputs = processor(
    text=candidate_labels, images=image, return_tensors="pt", padding=True
)
outputs = model(**inputs)

# logits_per_image: [1, num_labels] -> التشابه بين الصورة وكل نص مرشح
probs = outputs.logits_per_image.softmax(dim=1)
for label, prob in zip(candidate_labels, probs[0].tolist()):
    print(f"{label}: {prob:.2%}")

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-08-multimodal-models/chapter-40.md
# ────────────────────────────────────────────────────────────────────────
def build_image_index(image_paths: list[str], model, processor) -> torch.Tensor:
    """
    مرّر كل الصور عبر Image Encoder، استخرج تمثيلاتها المطبّعة،
    واحفظها في مصفوفة فهرس واحدة [num_images, embed_dim].
    """
    ...

def search_by_text(query: str, image_index: torch.Tensor, image_paths: list[str],
                    model, processor, top_k: int = 5) -> list[tuple[str, float]]:
    """
    حوّل الاستعلام النصي إلى تمثيل، احسب التشابه مع كل الصور المفهرسة،
    أعد أفضل top_k صورة مع درجة تشابهها.
    """
    ...

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-40-chapter-40.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 40: الفصل 40: CLIP — عندما تتعلّم الصورة والكلمة لغة مشتركة."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def cosine(left: list[float], right: list[float]) -> float:
    denominator = math.sqrt(sum((x * x for x in left)) * sum((y * y for y in right)))
    return sum((x * y for x, y in zip(left, right))) / denominator if denominator else 0.0

def run_clip() -> None:
    images = [[1.0, 0.0], [0.0, 1.0]]
    texts = [[0.9, 0.1], [0.1, 0.9]]
    scores = [[cosine(image, text) / 0.07 for text in texts] for image in images]
    print('image-text similarity logits:')
    for row in scores:
        print(' ', [round(x, 2) for x in row])
    print('matched-pair retrieval:', [max(range(2), key=row.__getitem__) for row in scores])

if __name__ == "__main__":
    print("Chapter 40 lab: Rank images with a shared-space CLIP similarity")
    run_clip()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-40-chapter-40.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
CLIP مصغّر من الصفر — تدريب تبايني (Contrastive) بين صور ونصوص
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

# أسماء فئات Fashion-MNIST — ستكون "المفردات" النصية لدينا
CLASS_NAMES = [
    "تي شيرت", "بنطال", "بلوفر", "فستان", "معطف",
    "صندل", "قميص", "حذاء رياضي", "حقيبة", "حذاء كاحل",
]

# قاموس حروف بسيط لترميز النصوص العربية حرفيًا (توضيحي وبسيط للتجربة)
VOCAB = sorted(set("".join(CLASS_NAMES) + " صورةهذهمنفئة"))
CHAR_TO_ID = {ch: i + 1 for i, ch in enumerate(VOCAB)}  # 0 محجوز للـ padding
VOCAB_SIZE = len(VOCAB) + 1
MAX_TEXT_LEN = 24


def encode_text(text: str) -> torch.Tensor:
    ids = [CHAR_TO_ID.get(ch, 0) for ch in text[:MAX_TEXT_LEN]]
    ids += [0] * (MAX_TEXT_LEN - len(ids))
    return torch.tensor(ids, dtype=torch.long)


# ---------------------------------------------------------------------
# 1) مجموعة بيانات (صورة، نص) مبنية من Fashion-MNIST
# ---------------------------------------------------------------------
class ImageTextDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        caption = f"صورة من فئة {CLASS_NAMES[label]}"
        text_ids = encode_text(caption)
        return image, text_ids, label


# ---------------------------------------------------------------------
# 2) مشفّر الصور (Image Encoder) — CNN بسيط (بديل مبسّط عن ViT)
# ---------------------------------------------------------------------
class ImageEncoder(nn.Module):
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 28 -> 14
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 14 -> 7
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Linear(128, embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.conv_layers(images).flatten(1)
        return self.projection(features)


# ---------------------------------------------------------------------
# 3) مشفّر النصوص (Text Encoder) — Transformer نصي بسيط
# ---------------------------------------------------------------------
class TextEncoder(nn.Module):
    def __init__(
        self, vocab_size: int, embed_dim: int = 128, num_heads: int = 4, depth: int = 2
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, MAX_TEXT_LEN, embed_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, text_ids: torch.Tensor) -> torch.Tensor:
        padding_mask = text_ids == 0
        x = self.token_embedding(text_ids) + self.position_embedding
        x = self.encoder(x, src_key_padding_mask=padding_mask)

        # استخلاص تمثيل أول رمز غير محشو كـ "ملخّص" للجملة (مشابه لـ [EOS] pooling)
        lengths = (~padding_mask).sum(dim=1) - 1
        lengths = lengths.clamp(min=0)
        pooled = x[torch.arange(x.size(0)), lengths]
        return self.projection(pooled)


# ---------------------------------------------------------------------
# 4) نموذج CLIP الكامل
# ---------------------------------------------------------------------
class MiniCLIP(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder = TextEncoder(vocab_size, embed_dim)
        # معامل درجة الحرارة القابل للتعلّم — يبدأ بما يوافق 1/0.07 كما في CLIP الأصلي
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        features = self.image_encoder(images)
        return F.normalize(features, dim=-1)

    def encode_text(self, text_ids: torch.Tensor) -> torch.Tensor:
        features = self.text_encoder(text_ids)
        return F.normalize(features, dim=-1)

    def forward(self, images: torch.Tensor, text_ids: torch.Tensor):
        image_features = self.encode_image(images)
        text_features = self.encode_text(text_ids)

        # تقييد logit_scale لتفادي الانفجار العددي (كما في CLIP الأصلي، يُقيَّد بـ 100)
        logit_scale = self.logit_scale.exp().clamp(max=100.0)

        # مصفوفة التشابه N x N
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text


# ---------------------------------------------------------------------
# 5) دالة خسارة CLIP التبايني المتناظر (Symmetric InfoNCE)
# ---------------------------------------------------------------------
def clip_loss(logits_per_image: torch.Tensor, logits_per_text: torch.Tensor) -> torch.Tensor:
    batch_size = logits_per_image.size(0)
    # العلامات الصحيحة هي ببساطة القطر: العنصر i يطابق العنصر i
    ground_truth = torch.arange(batch_size, device=logits_per_image.device)

    loss_image_to_text = F.cross_entropy(logits_per_image, ground_truth)
    loss_text_to_image = F.cross_entropy(logits_per_text, ground_truth)

    return (loss_image_to_text + loss_text_to_image) / 2.0


# ---------------------------------------------------------------------
# 6) حلقة التدريب
# ---------------------------------------------------------------------
def train_clip(model: nn.Module, train_loader: DataLoader, num_epochs: int = 10):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, text_ids, _ in train_loader:
            images, text_ids = images.to(device), text_ids.to(device)

            optimizer.zero_grad()
            logits_per_image, logits_per_text = model(images, text_ids)
            loss = clip_loss(logits_per_image, logits_per_text)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # تقييد logit_scale يدويًا بعد كل خطوة (ممارسة قياسية في CLIP)
            with torch.no_grad():
                model.logit_scale.clamp_(max=math.log(100.0))

            running_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch+1:02d}/{num_epochs} | Loss: {running_loss/len(train_loader):.4f}")


# ---------------------------------------------------------------------
# 7) الاستدلال: Zero-Shot Classification
# ---------------------------------------------------------------------
@torch.no_grad()
def zero_shot_classify(model: MiniCLIP, images: torch.Tensor, class_names: list[str]) -> torch.Tensor:
    """
    صنّف صورًا دون أي تدريب إضافي: حوّل كل فئة إلى جملة نصية،
    قارن تمثيل كل صورة بتمثيل كل جملة، واختر الأعلى تشابهًا.
    """
    model.eval()
    prompts = [f"صورة من فئة {name}" for name in class_names]
    text_ids = torch.stack([encode_text(p) for p in prompts]).to(device)

    text_features = model.encode_text(text_ids)        # [num_classes, D]
    image_features = model.encode_image(images.to(device))  # [B, D]

    similarity = image_features @ text_features.t()     # [B, num_classes]
    predictions = similarity.argmax(dim=-1)
    return predictions


# ---------------------------------------------------------------------
# 8) التشغيل الكامل
# ---------------------------------------------------------------------
if __name__ == "__main__":
    transform = T.Compose([T.ToTensor()])

    train_base = torchvision.datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_base = torchvision.datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_dataset = ImageTextDataset(train_base)
    test_dataset = ImageTextDataset(test_base)

    train_loader = DataLoader(
        train_dataset, batch_size=256, shuffle=True, num_workers=2, drop_last=True
    )

    model = MiniCLIP(vocab_size=VOCAB_SIZE, embed_dim=128)
    train_clip(model, train_loader, num_epochs=10)

    # تقييم Zero-Shot على مجموعة الاختبار
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    correct, total = 0, 0
    for images, _, labels in test_loader:
        predictions = zero_shot_classify(model, images, CLASS_NAMES)
        correct += (predictions.cpu() == labels).sum().item()
        total += labels.size(0)

    print(f"\nدقة Zero-Shot Classification: {correct/total:.2%}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-40-chapter-40.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = Image.open("cat.jpg")
candidate_labels = ["a photo of a cat", "a photo of a dog", "a photo of a car"]

inputs = processor(
    text=candidate_labels, images=image, return_tensors="pt", padding=True
)
outputs = model(**inputs)

# logits_per_image: [1, num_labels] -> التشابه بين الصورة وكل نص مرشح
probs = outputs.logits_per_image.softmax(dim=1)
for label, prob in zip(candidate_labels, probs[0].tolist()):
    print(f"{label}: {prob:.2%}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-40-chapter-40.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def build_image_index(image_paths: list[str], model, processor) -> torch.Tensor:
    """
    مرّر كل الصور عبر Image Encoder، استخرج تمثيلاتها المطبّعة،
    واحفظها في مصفوفة فهرس واحدة [num_images, embed_dim].
    """
    ...

def search_by_text(query: str, image_index: torch.Tensor, image_paths: list[str],
                    model, processor, top_k: int = 5) -> list[tuple[str, float]]:
    """
    حوّل الاستعلام النصي إلى تمثيل، احسب التشابه مع كل الصور المفهرسة،
    أعد أفضل top_k صورة مع درجة تشابهها.
    """
    ...
