#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — التعلّم الذاتي والنماذج التأسيسية
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 18 — Deep Dive into PyTorch

Book slug : pytorch-keras-deep-learning
Chapter   : ch-18-ch18 (lesson 18)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/pytorch-keras-deep-learning/code/ch-18-ch18.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — التعلّم الذاتي والنماذج التأسيسية
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 18 — Deep Dive into PyTorch

Book slug : pytorch-keras-deep-learning
Chapter   : ch-18-ch18 (lesson 18)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] src/chapters/ch18.html
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimCLR(nn.Module):
    """نموذج تعلّم بالتباين: عمود ترميز + رأس إسقاط."""

    def __init__(self, backbone, proj_dim=128):
        super().__init__()
        self.backbone = backbone
        self.projector = nn.Sequential(
            nn.Linear(backbone.output_dim, 512),
            nn.ReLU(),
            nn.Linear(512, proj_dim),
        )

    def forward(self, x1, x2):
        z1 = F.normalize(self.projector(self.backbone(x1)), dim=1)
        z2 = F.normalize(self.projector(self.backbone(x2)), dim=1)
        return z1, z2


def nt_xent_loss(z1, z2, temperature=0.1):
    """خسارة InfoNCE على دفعتين متزامنتين."""
    n = z1.size(0)
    z = torch.cat([z1, z2], dim=0)                 # (2n, d)
    sim = z @ z.t() / temperature
    sim.fill_diagonal_(-1e9)                        # استبعاد التمثيل نفسه
    targets = torch.arange(n, device=z1.device)
    targets = torch.cat([targets + n, targets])     # كل صورة ترى نسختها موجبة
    return F.cross_entropy(sim, targets)

# الناتج: tensor(0.6931) عند بداية التدريب ≈ ln(2)

# ────────────────────────────────────────────────────────────────────────
# [002] src/chapters/ch18.html
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

class MaskedImageEncoder(nn.Module):
    """يغطّي رقعاً عشوائية ثم يطلب من العمود استعادتها."""

    def __init__(self, patch_size=16, img_size=224, dim=192, mask_ratio=0.75):
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        n_patches = (img_size // patch_size) ** 2
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, 4, batch_first=True), num_layers=4
        )

    def forward(self, x):
        patches = self.patch_embed(x).flatten(2).transpose(1, 2)
        n = patches.size(1)
        keep = int(n * (1 - self.mask_ratio))
        visible = patches[:, :keep]
        masked = self.mask_token.expand(x.size(0), n - keep, -1)
        tokens = torch.cat([visible, masked], dim=1)
        out = self.encoder(tokens)
        return out[:, keep:]          # تمثيل الرقع المخفيّة فقط

# ────────────────────────────────────────────────────────────────────────
# [003] src/chapters/ch18.html
# ────────────────────────────────────────────────────────────────────────
import torch
from torch import nn

class FoundationModelAdapter(nn.Module):
    """رأس خفيف فوق نموذج تأسيسي مجمّد."""

    def __init__(self, foundation, num_classes=20, hidden=256):
        super().__init__()
        for p in foundation.parameters():
            p.requires_grad = False
        self.foundation = foundation
        dim = foundation.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.foundation(input_ids=input_ids,
                                   attention_mask=attention_mask).last_hidden_state
        pooled = out[:, 0]          # تمثيل الرمز الأول
        return self.head(pooled)

# الناتج: logits shape (batch, 20)

# ────────────────────────────────────────────────────────────────────────
# [004] src/chapters/ch18.html
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F


@torch.no_grad()
def embed_dataset(model, loader, device="cpu"):
    """يحسب تمثيل كلّ عيّنة ويُرجع مصفوفة (n, d) مع تسمياتها."""
    model.eval()
    vectors, labels = [], []
    for x, y in loader:
        x = x.to(device)
        emb = model.backbone(x) if hasattr(model, "backbone") else model(x)
        emb = F.normalize(emb, dim=1)
        vectors.append(emb.cpu())
        labels.append(y)
    return torch.cat(vectors), torch.cat(labels)


def knn_search(query_emb, matrix, k=10):
    """أقرب k جيران بقياس جيب التمام (الضرب النقطي على متجهات مطبَّعة)."""
    sims = query_emb @ matrix.T
    top_values, top_indices = sims.topk(k, dim=-1)
    return top_indices, top_values

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-18-ch18.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimCLR(nn.Module):
    """نموذج تعلّم بالتباين: عمود ترميز + رأس إسقاط."""

    def __init__(self, backbone, proj_dim=128):
        super().__init__()
        self.backbone = backbone
        self.projector = nn.Sequential(
            nn.Linear(backbone.output_dim, 512),
            nn.ReLU(),
            nn.Linear(512, proj_dim),
        )

    def forward(self, x1, x2):
        z1 = F.normalize(self.projector(self.backbone(x1)), dim=1)
        z2 = F.normalize(self.projector(self.backbone(x2)), dim=1)
        return z1, z2


def nt_xent_loss(z1, z2, temperature=0.1):
    """خسارة InfoNCE على دفعتين متزامنتين."""
    n = z1.size(0)
    z = torch.cat([z1, z2], dim=0)                 # (2n, d)
    sim = z @ z.t() / temperature
    sim.fill_diagonal_(-1e9)                        # استبعاد التمثيل نفسه
    targets = torch.arange(n, device=z1.device)
    targets = torch.cat([targets + n, targets])     # كل صورة ترى نسختها موجبة
    return F.cross_entropy(sim, targets)

# الناتج: tensor(0.6931) عند بداية التدريب ≈ ln(2)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-18-ch18.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

class MaskedImageEncoder(nn.Module):
    """يغطّي رقعاً عشوائية ثم يطلب من العمود استعادتها."""

    def __init__(self, patch_size=16, img_size=224, dim=192, mask_ratio=0.75):
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        n_patches = (img_size // patch_size) ** 2
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, 4, batch_first=True), num_layers=4
        )

    def forward(self, x):
        patches = self.patch_embed(x).flatten(2).transpose(1, 2)
        n = patches.size(1)
        keep = int(n * (1 - self.mask_ratio))
        visible = patches[:, :keep]
        masked = self.mask_token.expand(x.size(0), n - keep, -1)
        tokens = torch.cat([visible, masked], dim=1)
        out = self.encoder(tokens)
        return out[:, keep:]          # تمثيل الرقع المخفيّة فقط

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-18-ch18.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch
from torch import nn

class FoundationModelAdapter(nn.Module):
    """رأس خفيف فوق نموذج تأسيسي مجمّد."""

    def __init__(self, foundation, num_classes=20, hidden=256):
        super().__init__()
        for p in foundation.parameters():
            p.requires_grad = False
        self.foundation = foundation
        dim = foundation.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.foundation(input_ids=input_ids,
                                   attention_mask=attention_mask).last_hidden_state
        pooled = out[:, 0]          # تمثيل الرمز الأول
        return self.head(pooled)

# الناتج: logits shape (batch, 20)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-18-ch18.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F


@torch.no_grad()
def embed_dataset(model, loader, device="cpu"):
    """يحسب تمثيل كلّ عيّنة ويُرجع مصفوفة (n, d) مع تسمياتها."""
    model.eval()
    vectors, labels = [], []
    for x, y in loader:
        x = x.to(device)
        emb = model.backbone(x) if hasattr(model, "backbone") else model(x)
        emb = F.normalize(emb, dim=1)
        vectors.append(emb.cpu())
        labels.append(y)
    return torch.cat(vectors), torch.cat(labels)


def knn_search(query_emb, matrix, k=10):
    """أقرب k جيران بقياس جيب التمام (الضرب النقطي على متجهات مطبَّعة)."""
    sims = query_emb @ matrix.T
    top_values, top_indices = sims.topk(k, dim=-1)
    return top_indices, top_values
