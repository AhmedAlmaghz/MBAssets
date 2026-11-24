#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — الانتباه والمحوّلات Transformers
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 15 — Transformers

Book slug : pytorch-keras-deep-learning
Chapter   : ch-15-ch15 (lesson 15)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/pytorch-keras-deep-learning/code/ch-15-ch15.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — الانتباه والمحوّلات Transformers
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 15 — Transformers

Book slug : pytorch-keras-deep-learning
Chapter   : ch-15-ch15 (lesson 15)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] src/chapters/ch15.html
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

class AttentionBlock(nn.Module):
    """كتلة محوّل قياسية: انتباه متعدّد الرؤوس + شبكة تغذية أمامية."""

    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # اتّصال التخطّي حول الانتباه (Pre-LN)
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + self.drop(attn_out)

        # اتّصال التخطّي حول شبكة التغذية
        h = self.norm2(x)
        x = x + self.drop(self.ff(h))
        return x

# الناتج: tensor([[ 0.1234, -0.5678, ...]])
block = AttentionBlock(d_model=64, n_heads=4, d_ff=128)
x = torch.randn(2, 10, 64)                 # (دفعة، طول، بُعد)
print(block(x).shape)                      # torch.Size([2, 10, 64])

# ────────────────────────────────────────────────────────────────────────
# [002] src/chapters/ch15.html
# ────────────────────────────────────────────────────────────────────────
import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """ترميز موضعي جيبي: كل موضع يحصل على متجه فريد من جيب الزاوية."""

    def __init__(self, d_model=512, max_len=10_000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]

enc = PositionalEncoding(d_model=64)
x = torch.randn(2, 10, 64)
print(enc(x).shape)      # torch.Size([2, 10, 64])

# ────────────────────────────────────────────────────────────────────────
# [003] src/chapters/ch15.html
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """يحوّل صورة إلى تسلسل رموز، تماماً كما يفعل الإنسان."""

    def __init__(self, img_size=224, patch_size=16, in_ch=3, d_model=192):
        super().__init__()
        n_patches = (img_size // patch_size) ** 2
        self.patch = nn.Conv2d(in_ch, d_model, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, n_patches + 1, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):
        patches = self.patch(x).flatten(2).transpose(1, 2)   # (B, N, D)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, patches], dim=1)
        return x + self.pos

emb = PatchEmbedding(img_size=224, patch_size=16, d_model=192)
img = torch.randn(2, 3, 224, 224)
print(emb(img).shape)     # torch.Size([2, 197, 192])

# ────────────────────────────────────────────────────────────────────────
# [004] src/chapters/ch15.html
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """طبقة خطية مع تعديل منخفض الرتبة قابل للتعلّم."""

    def __init__(self, base: nn.Linear, rank=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank = rank
        self.scaling = alpha / rank
        self.A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)   # B يبدأ بصفر عمداً

    def forward(self, x):
        delta = (x @ self.A.T) @ self.B.T
        return self.base(x) + self.scaling * delta

# نجمّل طبقة داخل نموذج مدرَّب مسبقاً
model = nn.Sequential(nn.Linear(512, 512), nn.ReLU())
model[0] = LoRALinear(model[0], rank=8, alpha=16)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"trainable {trainable} / total {total}")
# الناتج: trainable 8192 / total 524800

# ────────────────────────────────────────────────────────────────────────
# [005] src/chapters/ch15.html
# ────────────────────────────────────────────────────────────────────────
import math
import torch

torch.manual_seed(0)
Q = torch.randn(3, 4)
K = torch.randn(3, 4)
V = torch.randn(3, 4)

dk = Q.shape[-1]
scores = Q @ K.T / math.sqrt(dk)
weights = scores.softmax(dim=-1)
out = weights @ V

print(scores)
print(weights)          # كل صفّ مجموعه 1
print(weights.sum(-1))  # tensor([1., 1., 1.])

# ────────────────────────────────────────────────────────────────────────
# [006] src/chapters/ch15.html
# ────────────────────────────────────────────────────────────────────────
import torch

def causal_mask(seq_len):
    """مصفوفة قنعة سببية: كل موضع يرى نفسه وما قبله فقط."""
    return torch.triu(
        torch.full((seq_len, seq_len), float("-inf")), diagonal=1
    )

mask = causal_mask(5)
print(mask[0])   # tensor([0., -inf, -inf, -inf, -inf])

attn = torch.nn.MultiheadAttention(8, 2, batch_first=True)
x = torch.randn(1, 5, 8)
y1, _ = attn(x, x, x, attn_mask=mask, need_weights=False)

x2 = x.clone()
x2[0, 3] = torch.randn(8)          # نغيّر كلمة في المنتصف
y2, _ = attn(x2, x2, x2, attn_mask=mask, need_weights=False)

print(torch.allclose(y1[0, :3], y2[0, :3]))   # True: ما قبلها لم يتغيّر

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-15-ch15.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

class AttentionBlock(nn.Module):
    """كتلة محوّل قياسية: انتباه متعدّد الرؤوس + شبكة تغذية أمامية."""

    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # اتّصال التخطّي حول الانتباه (Pre-LN)
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + self.drop(attn_out)

        # اتّصال التخطّي حول شبكة التغذية
        h = self.norm2(x)
        x = x + self.drop(self.ff(h))
        return x

# الناتج: tensor([[ 0.1234, -0.5678, ...]])
block = AttentionBlock(d_model=64, n_heads=4, d_ff=128)
x = torch.randn(2, 10, 64)                 # (دفعة، طول، بُعد)
print(block(x).shape)                      # torch.Size([2, 10, 64])

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-15-ch15.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """ترميز موضعي جيبي: كل موضع يحصل على متجه فريد من جيب الزاوية."""

    def __init__(self, d_model=512, max_len=10_000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]

enc = PositionalEncoding(d_model=64)
x = torch.randn(2, 10, 64)
print(enc(x).shape)      # torch.Size([2, 10, 64])

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-15-ch15.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """يحوّل صورة إلى تسلسل رموز، تماماً كما يفعل الإنسان."""

    def __init__(self, img_size=224, patch_size=16, in_ch=3, d_model=192):
        super().__init__()
        n_patches = (img_size // patch_size) ** 2
        self.patch = nn.Conv2d(in_ch, d_model, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, n_patches + 1, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):
        patches = self.patch(x).flatten(2).transpose(1, 2)   # (B, N, D)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, patches], dim=1)
        return x + self.pos

emb = PatchEmbedding(img_size=224, patch_size=16, d_model=192)
img = torch.randn(2, 3, 224, 224)
print(emb(img).shape)     # torch.Size([2, 197, 192])

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-15-ch15.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """طبقة خطية مع تعديل منخفض الرتبة قابل للتعلّم."""

    def __init__(self, base: nn.Linear, rank=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank = rank
        self.scaling = alpha / rank
        self.A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)   # B يبدأ بصفر عمداً

    def forward(self, x):
        delta = (x @ self.A.T) @ self.B.T
        return self.base(x) + self.scaling * delta

# نجمّل طبقة داخل نموذج مدرَّب مسبقاً
model = nn.Sequential(nn.Linear(512, 512), nn.ReLU())
model[0] = LoRALinear(model[0], rank=8, alpha=16)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"trainable {trainable} / total {total}")
# الناتج: trainable 8192 / total 524800

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-15-ch15.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import math
import torch

torch.manual_seed(0)
Q = torch.randn(3, 4)
K = torch.randn(3, 4)
V = torch.randn(3, 4)

dk = Q.shape[-1]
scores = Q @ K.T / math.sqrt(dk)
weights = scores.softmax(dim=-1)
out = weights @ V

print(scores)
print(weights)          # كل صفّ مجموعه 1
print(weights.sum(-1))  # tensor([1., 1., 1.])

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-15-ch15.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import torch

def causal_mask(seq_len):
    """مصفوفة قنعة سببية: كل موضع يرى نفسه وما قبله فقط."""
    return torch.triu(
        torch.full((seq_len, seq_len), float("-inf")), diagonal=1
    )

mask = causal_mask(5)
print(mask[0])   # tensor([0., -inf, -inf, -inf, -inf])

attn = torch.nn.MultiheadAttention(8, 2, batch_first=True)
x = torch.randn(1, 5, 8)
y1, _ = attn(x, x, x, attn_mask=mask, need_weights=False)

x2 = x.clone()
x2[0, 3] = torch.randn(8)          # نغيّر كلمة في المنتصف
y2, _ = attn(x2, x2, x2, attn_mask=mask, need_weights=False)

print(torch.allclose(y1[0, :3], y2[0, :3]))   # True: ما قبلها لم يتغيّر
