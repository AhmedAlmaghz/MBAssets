#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — Vision Transformer (ViT) — من البكسل إلى الرمز
Large Language Model Engineering — Vision Transformer (ViT)

Book slug : llm-engineering-69-chapters
Chapter   : ch-39-chapter-39 (lesson 39)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-39-chapter-39.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — Vision Transformer (ViT) — من البكسل إلى الرمز
Large Language Model Engineering — Vision Transformer (ViT)

Book slug : llm-engineering-69-chapters
Chapter   : ch-39-chapter-39 (lesson 39)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-39-chapter-39.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — Vision Transformer (ViT) — من البكسل إلى الرمز
Large Language Model Engineering — Vision Transformer (ViT)

Book slug : llm-engineering-69-chapters
Chapter   : ch-39-chapter-39 (lesson 39)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_39.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 39: الفصل 39: Vision Transformer (ViT) — من البكسل إلى الرمز."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_vision_patches() -> None:
    image = [[row * 4 + col for col in range(4)] for row in range(4)]
    patch_size = 2
    patches = [[image[r + dr][c + dc] for dr in range(patch_size) for dc in range(patch_size)] for r in range(0, 4, patch_size) for c in range(0, 4, patch_size)]
    print('image / patch:', (4, 4), '/', (patch_size, patch_size))
    print('patch token count:', len(patches))
    print('flattened patch vectors:', patches)

if __name__ == "__main__":
    print("Chapter 39 lab: Split an image into Vision Transformer patches")
    run_vision_patches()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-08-multimodal-models/chapter-39.md
# ────────────────────────────────────────────────────────────────────────
"""
Vision Transformer (ViT) من الصفر — تدريب على CIFAR-10
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# 1) إعدادات المعمارية
# ---------------------------------------------------------------------
@dataclass
class ViTConfig:
    image_size: int = 32          # أبعاد الصورة (CIFAR-10 أصغر من 224)
    patch_size: int = 4           # حجم الرقعة
    num_channels: int = 3
    num_classes: int = 10
    embed_dim: int = 256
    depth: int = 6                # عدد طبقات Transformer
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2


# ---------------------------------------------------------------------
# 2) طبقة تضمين الرقع (Patch Embedding) عبر Convolution
# ---------------------------------------------------------------------
class PatchEmbedding(nn.Module):
    """
    تحويل صورة [B, C, H, W] إلى تسلسل رقع مُضمَّنة [B, N, D].
    نستخدم Conv2d بـ kernel=stride=patch_size — هذا مكافئ رياضيًا
    لتسطيح كل رقعة ثم إسقاطها خطيًا، لكن أسرع على GPU.
    """

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> [B, D, H/P, W/P]
        x = self.projection(x)
        # تسطيح الأبعاد المكانية ثم نقل المحاور: [B, D, N] -> [B, N, D]
        x = x.flatten(2).transpose(1, 2)
        return x


# ---------------------------------------------------------------------
# 3) كتلة Transformer Encoder (Pre-LN) — نفس معمارية BERT/GPT
# ---------------------------------------------------------------------
class TransformerEncoderBlock(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(config.embed_dim)
        hidden_dim = int(config.embed_dim * config.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, config.embed_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN: التطبيع قبل Attention/MLP، مع Residual Connection
        normed = self.norm1(x)
        attn_out, _ = self.attention(normed, normed, normed, need_weights=False)
        x = x + attn_out

        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------
# 4) معمارية ViT الكاملة
# ---------------------------------------------------------------------
class VisionTransformer(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbedding(config)

        # رمز [CLS] قابل للتعلّم — يُستخدم تمثيله النهائي للتصنيف
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))

        # تضمينات موضعية قابلة للتعلّم (+1 لأجل رمز CLS)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.num_patches + 1, config.embed_dim)
        )

        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [TransformerEncoderBlock(config) for _ in range(config.depth)]
        )

        self.norm = nn.LayerNorm(config.embed_dim)
        self.classification_head = nn.Linear(config.embed_dim, config.num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        # 1. تحويل الصورة إلى تسلسل رقع مُضمَّنة
        patches = self.patch_embed(x)  # [B, N, D]

        # 2. إضافة رمز [CLS] في البداية
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, D]
        tokens = torch.cat([cls_tokens, patches], dim=1)        # [B, N+1, D]

        # 3. إضافة التضمينات الموضعية
        tokens = tokens + self.position_embedding
        tokens = self.dropout(tokens)

        # 4. تمرير عبر طبقات Transformer
        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)

        # 5. استخلاص تمثيل [CLS] فقط للتصنيف
        cls_representation = tokens[:, 0]
        logits = self.classification_head(cls_representation)
        return logits


# ---------------------------------------------------------------------
# 5) تجهيز بيانات CIFAR-10
# ---------------------------------------------------------------------
def build_dataloaders(batch_size: int = 128):
    train_transform = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
        ]
    )
    test_transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
        ]
    )

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2
    )
    return train_loader, test_loader


# ---------------------------------------------------------------------
# 6) حلقة التدريب والتقييم
# ---------------------------------------------------------------------
def evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            predictions = logits.argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return correct / total


def train_vit(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int = 20,
    learning_rate: float = 3e-4,
):
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.05
    )
    # جدولة Cosine شائعة جدًا في تدريب ViT
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        test_accuracy = evaluate(model, test_loader)
        print(
            f"Epoch {epoch+1:02d}/{num_epochs} | "
            f"Loss: {avg_loss:.4f} | Test Accuracy: {test_accuracy:.2%}"
        )


# ---------------------------------------------------------------------
# 7) التشغيل الكامل
# ---------------------------------------------------------------------
if __name__ == "__main__":
    config = ViTConfig(
        image_size=32,
        patch_size=4,
        num_classes=10,
        embed_dim=256,
        depth=6,
        num_heads=8,
    )

    model = VisionTransformer(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"عدد البارامترات: {num_params:,}")
    print(f"عدد الرقع لكل صورة: {config.num_patches}")

    train_loader, test_loader = build_dataloaders(batch_size=128)
    train_vit(model, train_loader, test_loader, num_epochs=20)

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-08-multimodal-models/chapter-39.md
# ────────────────────────────────────────────────────────────────────────
def extract_cls_attention_weights(model: VisionTransformer, image: torch.Tensor) -> torch.Tensor:
    """
    عدّل forward pass لاستخراج أوزان Attention من [CLS] إلى كل الرقع
    في الطبقة الأخيرة فقط.
    تلميح: ستحتاج لتعديل MultiheadAttention لإرجاع average_attn_weights=False
    أو need_weights=True عند الاستدعاء.
    """
    ...

def visualize_attention_map(image: torch.Tensor, attention_weights: torch.Tensor, patch_size: int):
    """
    أعد تشكيل أوزان الانتباه (196 قيمة مثلًا) إلى شبكة 2D (14×14)،
    كبّرها (Upsample) لتطابق حجم الصورة الأصلية،
    وارسمها كـ Heatmap فوق الصورة الأصلية باستخدام matplotlib.
    """
    ...

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-39-chapter-39.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 39: الفصل 39: Vision Transformer (ViT) — من البكسل إلى الرمز."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_vision_patches() -> None:
    image = [[row * 4 + col for col in range(4)] for row in range(4)]
    patch_size = 2
    patches = [[image[r + dr][c + dc] for dr in range(patch_size) for dc in range(patch_size)] for r in range(0, 4, patch_size) for c in range(0, 4, patch_size)]
    print('image / patch:', (4, 4), '/', (patch_size, patch_size))
    print('patch token count:', len(patches))
    print('flattened patch vectors:', patches)

if __name__ == "__main__":
    print("Chapter 39 lab: Split an image into Vision Transformer patches")
    run_vision_patches()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-39-chapter-39.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""
Vision Transformer (ViT) من الصفر — تدريب على CIFAR-10
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# 1) إعدادات المعمارية
# ---------------------------------------------------------------------
@dataclass
class ViTConfig:
    image_size: int = 32          # أبعاد الصورة (CIFAR-10 أصغر من 224)
    patch_size: int = 4           # حجم الرقعة
    num_channels: int = 3
    num_classes: int = 10
    embed_dim: int = 256
    depth: int = 6                # عدد طبقات Transformer
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2


# ---------------------------------------------------------------------
# 2) طبقة تضمين الرقع (Patch Embedding) عبر Convolution
# ---------------------------------------------------------------------
class PatchEmbedding(nn.Module):
    """
    تحويل صورة [B, C, H, W] إلى تسلسل رقع مُضمَّنة [B, N, D].
    نستخدم Conv2d بـ kernel=stride=patch_size — هذا مكافئ رياضيًا
    لتسطيح كل رقعة ثم إسقاطها خطيًا، لكن أسرع على GPU.
    """

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> [B, D, H/P, W/P]
        x = self.projection(x)
        # تسطيح الأبعاد المكانية ثم نقل المحاور: [B, D, N] -> [B, N, D]
        x = x.flatten(2).transpose(1, 2)
        return x


# ---------------------------------------------------------------------
# 3) كتلة Transformer Encoder (Pre-LN) — نفس معمارية BERT/GPT
# ---------------------------------------------------------------------
class TransformerEncoderBlock(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(config.embed_dim)
        hidden_dim = int(config.embed_dim * config.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, config.embed_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN: التطبيع قبل Attention/MLP، مع Residual Connection
        normed = self.norm1(x)
        attn_out, _ = self.attention(normed, normed, normed, need_weights=False)
        x = x + attn_out

        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------
# 4) معمارية ViT الكاملة
# ---------------------------------------------------------------------
class VisionTransformer(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbedding(config)

        # رمز [CLS] قابل للتعلّم — يُستخدم تمثيله النهائي للتصنيف
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))

        # تضمينات موضعية قابلة للتعلّم (+1 لأجل رمز CLS)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.num_patches + 1, config.embed_dim)
        )

        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [TransformerEncoderBlock(config) for _ in range(config.depth)]
        )

        self.norm = nn.LayerNorm(config.embed_dim)
        self.classification_head = nn.Linear(config.embed_dim, config.num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        # 1. تحويل الصورة إلى تسلسل رقع مُضمَّنة
        patches = self.patch_embed(x)  # [B, N, D]

        # 2. إضافة رمز [CLS] في البداية
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, D]
        tokens = torch.cat([cls_tokens, patches], dim=1)        # [B, N+1, D]

        # 3. إضافة التضمينات الموضعية
        tokens = tokens + self.position_embedding
        tokens = self.dropout(tokens)

        # 4. تمرير عبر طبقات Transformer
        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)

        # 5. استخلاص تمثيل [CLS] فقط للتصنيف
        cls_representation = tokens[:, 0]
        logits = self.classification_head(cls_representation)
        return logits


# ---------------------------------------------------------------------
# 5) تجهيز بيانات CIFAR-10
# ---------------------------------------------------------------------
def build_dataloaders(batch_size: int = 128):
    train_transform = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
        ]
    )
    test_transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
        ]
    )

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2
    )
    return train_loader, test_loader


# ---------------------------------------------------------------------
# 6) حلقة التدريب والتقييم
# ---------------------------------------------------------------------
def evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            predictions = logits.argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return correct / total


def train_vit(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int = 20,
    learning_rate: float = 3e-4,
):
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.05
    )
    # جدولة Cosine شائعة جدًا في تدريب ViT
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        test_accuracy = evaluate(model, test_loader)
        print(
            f"Epoch {epoch+1:02d}/{num_epochs} | "
            f"Loss: {avg_loss:.4f} | Test Accuracy: {test_accuracy:.2%}"
        )


# ---------------------------------------------------------------------
# 7) التشغيل الكامل
# ---------------------------------------------------------------------
if __name__ == "__main__":
    config = ViTConfig(
        image_size=32,
        patch_size=4,
        num_classes=10,
        embed_dim=256,
        depth=6,
        num_heads=8,
    )

    model = VisionTransformer(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"عدد البارامترات: {num_params:,}")
    print(f"عدد الرقع لكل صورة: {config.num_patches}")

    train_loader, test_loader = build_dataloaders(batch_size=128)
    train_vit(model, train_loader, test_loader, num_epochs=20)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-39-chapter-39.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def extract_cls_attention_weights(model: VisionTransformer, image: torch.Tensor) -> torch.Tensor:
    """
    عدّل forward pass لاستخراج أوزان Attention من [CLS] إلى كل الرقع
    في الطبقة الأخيرة فقط.
    تلميح: ستحتاج لتعديل MultiheadAttention لإرجاع average_attn_weights=False
    أو need_weights=True عند الاستدعاء.
    """
    ...

def visualize_attention_map(image: torch.Tensor, attention_weights: torch.Tensor, patch_size: int):
    """
    أعد تشكيل أوزان الانتباه (196 قيمة مثلًا) إلى شبكة 2D (14×14)،
    كبّرها (Upsample) لتطابق حجم الصورة الأصلية،
    وارسمها كـ Heatmap فوق الصورة الأصلية باستخدام matplotlib.
    """
    ...

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-39-chapter-39.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — Vision Transformer (ViT) — من البكسل إلى الرمز
Large Language Model Engineering — Vision Transformer (ViT)

Book slug : llm-engineering-69-chapters
Chapter   : ch-39-chapter-39 (lesson 39)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_39.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 39: الفصل 39: Vision Transformer (ViT) — من البكسل إلى الرمز."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_vision_patches() -> None:
    image = [[row * 4 + col for col in range(4)] for row in range(4)]
    patch_size = 2
    patches = [[image[r + dr][c + dc] for dr in range(patch_size) for dc in range(patch_size)] for r in range(0, 4, patch_size) for c in range(0, 4, patch_size)]
    print('image / patch:', (4, 4), '/', (patch_size, patch_size))
    print('patch token count:', len(patches))
    print('flattened patch vectors:', patches)

if __name__ == "__main__":
    print("Chapter 39 lab: Split an image into Vision Transformer patches")
    run_vision_patches()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-08-multimodal-models/chapter-39.md
# ────────────────────────────────────────────────────────────────────────
"""
Vision Transformer (ViT) من الصفر — تدريب على CIFAR-10
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# 1) إعدادات المعمارية
# ---------------------------------------------------------------------
@dataclass
class ViTConfig:
    image_size: int = 32          # أبعاد الصورة (CIFAR-10 أصغر من 224)
    patch_size: int = 4           # حجم الرقعة
    num_channels: int = 3
    num_classes: int = 10
    embed_dim: int = 256
    depth: int = 6                # عدد طبقات Transformer
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2


# ---------------------------------------------------------------------
# 2) طبقة تضمين الرقع (Patch Embedding) عبر Convolution
# ---------------------------------------------------------------------
class PatchEmbedding(nn.Module):
    """
    تحويل صورة [B, C, H, W] إلى تسلسل رقع مُضمَّنة [B, N, D].
    نستخدم Conv2d بـ kernel=stride=patch_size — هذا مكافئ رياضيًا
    لتسطيح كل رقعة ثم إسقاطها خطيًا، لكن أسرع على GPU.
    """

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> [B, D, H/P, W/P]
        x = self.projection(x)
        # تسطيح الأبعاد المكانية ثم نقل المحاور: [B, D, N] -> [B, N, D]
        x = x.flatten(2).transpose(1, 2)
        return x


# ---------------------------------------------------------------------
# 3) كتلة Transformer Encoder (Pre-LN) — نفس معمارية BERT/GPT
# ---------------------------------------------------------------------
class TransformerEncoderBlock(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(config.embed_dim)
        hidden_dim = int(config.embed_dim * config.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, config.embed_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN: التطبيع قبل Attention/MLP، مع Residual Connection
        normed = self.norm1(x)
        attn_out, _ = self.attention(normed, normed, normed, need_weights=False)
        x = x + attn_out

        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------
# 4) معمارية ViT الكاملة
# ---------------------------------------------------------------------
class VisionTransformer(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbedding(config)

        # رمز [CLS] قابل للتعلّم — يُستخدم تمثيله النهائي للتصنيف
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))

        # تضمينات موضعية قابلة للتعلّم (+1 لأجل رمز CLS)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.num_patches + 1, config.embed_dim)
        )

        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [TransformerEncoderBlock(config) for _ in range(config.depth)]
        )

        self.norm = nn.LayerNorm(config.embed_dim)
        self.classification_head = nn.Linear(config.embed_dim, config.num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        # 1. تحويل الصورة إلى تسلسل رقع مُضمَّنة
        patches = self.patch_embed(x)  # [B, N, D]

        # 2. إضافة رمز [CLS] في البداية
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, D]
        tokens = torch.cat([cls_tokens, patches], dim=1)        # [B, N+1, D]

        # 3. إضافة التضمينات الموضعية
        tokens = tokens + self.position_embedding
        tokens = self.dropout(tokens)

        # 4. تمرير عبر طبقات Transformer
        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)

        # 5. استخلاص تمثيل [CLS] فقط للتصنيف
        cls_representation = tokens[:, 0]
        logits = self.classification_head(cls_representation)
        return logits


# ---------------------------------------------------------------------
# 5) تجهيز بيانات CIFAR-10
# ---------------------------------------------------------------------
def build_dataloaders(batch_size: int = 128):
    train_transform = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
        ]
    )
    test_transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
        ]
    )

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2
    )
    return train_loader, test_loader


# ---------------------------------------------------------------------
# 6) حلقة التدريب والتقييم
# ---------------------------------------------------------------------
def evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            predictions = logits.argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return correct / total


def train_vit(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int = 20,
    learning_rate: float = 3e-4,
):
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.05
    )
    # جدولة Cosine شائعة جدًا في تدريب ViT
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        test_accuracy = evaluate(model, test_loader)
        print(
            f"Epoch {epoch+1:02d}/{num_epochs} | "
            f"Loss: {avg_loss:.4f} | Test Accuracy: {test_accuracy:.2%}"
        )


# ---------------------------------------------------------------------
# 7) التشغيل الكامل
# ---------------------------------------------------------------------
if __name__ == "__main__":
    config = ViTConfig(
        image_size=32,
        patch_size=4,
        num_classes=10,
        embed_dim=256,
        depth=6,
        num_heads=8,
    )

    model = VisionTransformer(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"عدد البارامترات: {num_params:,}")
    print(f"عدد الرقع لكل صورة: {config.num_patches}")

    train_loader, test_loader = build_dataloaders(batch_size=128)
    train_vit(model, train_loader, test_loader, num_epochs=20)

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-08-multimodal-models/chapter-39.md
# ────────────────────────────────────────────────────────────────────────
def extract_cls_attention_weights(model: VisionTransformer, image: torch.Tensor) -> torch.Tensor:
    """
    عدّل forward pass لاستخراج أوزان Attention من [CLS] إلى كل الرقع
    في الطبقة الأخيرة فقط.
    تلميح: ستحتاج لتعديل MultiheadAttention لإرجاع average_attn_weights=False
    أو need_weights=True عند الاستدعاء.
    """
    ...

def visualize_attention_map(image: torch.Tensor, attention_weights: torch.Tensor, patch_size: int):
    """
    أعد تشكيل أوزان الانتباه (196 قيمة مثلًا) إلى شبكة 2D (14×14)،
    كبّرها (Upsample) لتطابق حجم الصورة الأصلية،
    وارسمها كـ Heatmap فوق الصورة الأصلية باستخدام matplotlib.
    """
    ...

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-39-chapter-39.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 39: الفصل 39: Vision Transformer (ViT) — من البكسل إلى الرمز."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_vision_patches() -> None:
    image = [[row * 4 + col for col in range(4)] for row in range(4)]
    patch_size = 2
    patches = [[image[r + dr][c + dc] for dr in range(patch_size) for dc in range(patch_size)] for r in range(0, 4, patch_size) for c in range(0, 4, patch_size)]
    print('image / patch:', (4, 4), '/', (patch_size, patch_size))
    print('patch token count:', len(patches))
    print('flattened patch vectors:', patches)

if __name__ == "__main__":
    print("Chapter 39 lab: Split an image into Vision Transformer patches")
    run_vision_patches()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-39-chapter-39.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
Vision Transformer (ViT) من الصفر — تدريب على CIFAR-10
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# 1) إعدادات المعمارية
# ---------------------------------------------------------------------
@dataclass
class ViTConfig:
    image_size: int = 32          # أبعاد الصورة (CIFAR-10 أصغر من 224)
    patch_size: int = 4           # حجم الرقعة
    num_channels: int = 3
    num_classes: int = 10
    embed_dim: int = 256
    depth: int = 6                # عدد طبقات Transformer
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2


# ---------------------------------------------------------------------
# 2) طبقة تضمين الرقع (Patch Embedding) عبر Convolution
# ---------------------------------------------------------------------
class PatchEmbedding(nn.Module):
    """
    تحويل صورة [B, C, H, W] إلى تسلسل رقع مُضمَّنة [B, N, D].
    نستخدم Conv2d بـ kernel=stride=patch_size — هذا مكافئ رياضيًا
    لتسطيح كل رقعة ثم إسقاطها خطيًا، لكن أسرع على GPU.
    """

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> [B, D, H/P, W/P]
        x = self.projection(x)
        # تسطيح الأبعاد المكانية ثم نقل المحاور: [B, D, N] -> [B, N, D]
        x = x.flatten(2).transpose(1, 2)
        return x


# ---------------------------------------------------------------------
# 3) كتلة Transformer Encoder (Pre-LN) — نفس معمارية BERT/GPT
# ---------------------------------------------------------------------
class TransformerEncoderBlock(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(config.embed_dim)
        hidden_dim = int(config.embed_dim * config.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, config.embed_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN: التطبيع قبل Attention/MLP، مع Residual Connection
        normed = self.norm1(x)
        attn_out, _ = self.attention(normed, normed, normed, need_weights=False)
        x = x + attn_out

        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------
# 4) معمارية ViT الكاملة
# ---------------------------------------------------------------------
class VisionTransformer(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbedding(config)

        # رمز [CLS] قابل للتعلّم — يُستخدم تمثيله النهائي للتصنيف
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))

        # تضمينات موضعية قابلة للتعلّم (+1 لأجل رمز CLS)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.num_patches + 1, config.embed_dim)
        )

        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [TransformerEncoderBlock(config) for _ in range(config.depth)]
        )

        self.norm = nn.LayerNorm(config.embed_dim)
        self.classification_head = nn.Linear(config.embed_dim, config.num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        # 1. تحويل الصورة إلى تسلسل رقع مُضمَّنة
        patches = self.patch_embed(x)  # [B, N, D]

        # 2. إضافة رمز [CLS] في البداية
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, D]
        tokens = torch.cat([cls_tokens, patches], dim=1)        # [B, N+1, D]

        # 3. إضافة التضمينات الموضعية
        tokens = tokens + self.position_embedding
        tokens = self.dropout(tokens)

        # 4. تمرير عبر طبقات Transformer
        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)

        # 5. استخلاص تمثيل [CLS] فقط للتصنيف
        cls_representation = tokens[:, 0]
        logits = self.classification_head(cls_representation)
        return logits


# ---------------------------------------------------------------------
# 5) تجهيز بيانات CIFAR-10
# ---------------------------------------------------------------------
def build_dataloaders(batch_size: int = 128):
    train_transform = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
        ]
    )
    test_transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
        ]
    )

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2
    )
    return train_loader, test_loader


# ---------------------------------------------------------------------
# 6) حلقة التدريب والتقييم
# ---------------------------------------------------------------------
def evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            predictions = logits.argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return correct / total


def train_vit(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int = 20,
    learning_rate: float = 3e-4,
):
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.05
    )
    # جدولة Cosine شائعة جدًا في تدريب ViT
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        test_accuracy = evaluate(model, test_loader)
        print(
            f"Epoch {epoch+1:02d}/{num_epochs} | "
            f"Loss: {avg_loss:.4f} | Test Accuracy: {test_accuracy:.2%}"
        )


# ---------------------------------------------------------------------
# 7) التشغيل الكامل
# ---------------------------------------------------------------------
if __name__ == "__main__":
    config = ViTConfig(
        image_size=32,
        patch_size=4,
        num_classes=10,
        embed_dim=256,
        depth=6,
        num_heads=8,
    )

    model = VisionTransformer(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"عدد البارامترات: {num_params:,}")
    print(f"عدد الرقع لكل صورة: {config.num_patches}")

    train_loader, test_loader = build_dataloaders(batch_size=128)
    train_vit(model, train_loader, test_loader, num_epochs=20)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-39-chapter-39.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def extract_cls_attention_weights(model: VisionTransformer, image: torch.Tensor) -> torch.Tensor:
    """
    عدّل forward pass لاستخراج أوزان Attention من [CLS] إلى كل الرقع
    في الطبقة الأخيرة فقط.
    تلميح: ستحتاج لتعديل MultiheadAttention لإرجاع average_attn_weights=False
    أو need_weights=True عند الاستدعاء.
    """
    ...

def visualize_attention_map(image: torch.Tensor, attention_weights: torch.Tensor, patch_size: int):
    """
    أعد تشكيل أوزان الانتباه (196 قيمة مثلًا) إلى شبكة 2D (14×14)،
    كبّرها (Upsample) لتطابق حجم الصورة الأصلية،
    وارسمها كـ Heatmap فوق الصورة الأصلية باستخدام matplotlib.
    """
    ...
