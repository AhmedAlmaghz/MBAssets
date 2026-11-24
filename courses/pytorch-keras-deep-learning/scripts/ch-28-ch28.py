#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — دراسات حالة من الحياة العملية
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 28 — Engineering and Professional Practice

Book slug : pytorch-keras-deep-learning
Chapter   : ch-28-ch28 (lesson 28)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/pytorch-keras-deep-learning/code/ch-28-ch28.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch و Keras: الدليل الشامل في التعلم العميق — دراسات حالة من الحياة العملية
PyTorch and Keras: The Complete Deep Learning Guide — Chapter 28 — Engineering and Professional Practice

Book slug : pytorch-keras-deep-learning
Chapter   : ch-28-ch28 (lesson 28)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] src/chapters/ch28.html
# ────────────────────────────────────────────────────────────────────────
# نموذج الفرز: انتباه فوق خريطة سمات CNN
import torch
from torch import nn

class AttentionClassifier(nn.Module):
    def __init__(self, backbone, n_classes=11):
        super().__init__()
        self.backbone = backbone
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()              # نأخذ السمات لا المخرجات
        self.attention = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.head = nn.Linear(feat_dim, n_classes)

    def forward(self, x, return_attention=False):
        feats = self.backbone(x)                # (B, C)
        w = torch.softmax(self.attention(feats), dim=1)   # (B, C)
        pooled = (feats * w).sum(dim=1)         # تجميع مرجّح
        logits = self.head(pooled)
        return (logits, w) if return_attention else logits

# ────────────────────────────────────────────────────────────────────────
# [002] src/chapters/ch28.html
# ────────────────────────────────────────────────────────────────────────
# المزيج: نموذج شجري + شبكة عصبية (fusion)
import torch
from torch import nn

class HybridFraud(nn.Module):
    def __init__(self, tabular_model, seq_encoder, n_features=30):
        super().__init__()
        self.tab = tabular_model          # GBDT outputs (log-odds) كسمات
        self.seq = seq_encoder           # LSTM/Transformer على تاريخ المستخدم
        self.head = nn.Sequential(
            nn.Linear(64 + n_features, 64), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, seq_emb, tabular_feats):
        z = torch.cat([self.seq(seq_emb), tabular_feats], dim=1)
        return self.head(z).squeeze(-1)     # logits

# ────────────────────────────────────────────────────────────────────────
# [003] src/chapters/ch28.html
# ────────────────────────────────────────────────────────────────────────
# شبكة GRU مع mask للقنوات الناقصة (sensor dropouts)
import torch
from torch import nn

class SensorGRU(nn.Module):
    def __init__(self, n_sensors=12, hidden=64, n_out=1):
        super().__init__()
        self.gru = nn.GRU(n_sensors, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_out)

    def forward(self, x, mask=None):
        # x: (B, T, n_sensors)   mask: (B, T) 1 للقناة المتاحة
        lengths = mask.sum(dim=1).clamp(min=1) if mask is not None else None
        out, _ = self.gru(x)
        if lengths is not None:
            idx = (lengths - 1).long()
            pooled = out[torch.arange(out.size(0)), idx]
        else:
            pooled = out[:, -1]
        return self.head(pooled).squeeze(-1)

# ────────────────────────────────────────────────────────────────────────
# [004] src/chapters/ch28.html
# ────────────────────────────────────────────────────────────────────────
# بنية RAG مختصرة: استرجاع ثم توليد مع اقتباسات
def answer(question, retriever, generator, k=5, score_threshold=0.35):
    hits = retriever.search(question, k=k)
    if not hits or hits[0]["score"] < score_threshold:
        return {"answer": "لا أعرف من الوثائق المتوفّرة.",
                "citations": [], "confidence": "low"}

    context = "\n\n".join(
        f"[{i+1}] {h['text']} (المصدر: {h['source']})"
        for i, h in enumerate(hits)
    )
    prompt = (
        "أجب عن السؤال اعتماداً على المقاطع التالية فقط. "
        "إذا لم تكفِ المقاطع، أجب: لا أعلم.\n\n"
        f"المقاطع:\n{context}\n\nالسؤال: {question}\nالإجابة:"
    )
    text = generator.generate(prompt)
    return {"answer": text,
            "citations": [h["source"] for h in hits],
            "confidence": "high" if hits[0]["score"] > 0.6 else "medium"}

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-28-ch28.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# نموذج الفرز: انتباه فوق خريطة سمات CNN
import torch
from torch import nn

class AttentionClassifier(nn.Module):
    def __init__(self, backbone, n_classes=11):
        super().__init__()
        self.backbone = backbone
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()              # نأخذ السمات لا المخرجات
        self.attention = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.head = nn.Linear(feat_dim, n_classes)

    def forward(self, x, return_attention=False):
        feats = self.backbone(x)                # (B, C)
        w = torch.softmax(self.attention(feats), dim=1)   # (B, C)
        pooled = (feats * w).sum(dim=1)         # تجميع مرجّح
        logits = self.head(pooled)
        return (logits, w) if return_attention else logits

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-28-ch28.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# المزيج: نموذج شجري + شبكة عصبية (fusion)
import torch
from torch import nn

class HybridFraud(nn.Module):
    def __init__(self, tabular_model, seq_encoder, n_features=30):
        super().__init__()
        self.tab = tabular_model          # GBDT outputs (log-odds) كسمات
        self.seq = seq_encoder           # LSTM/Transformer على تاريخ المستخدم
        self.head = nn.Sequential(
            nn.Linear(64 + n_features, 64), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, seq_emb, tabular_feats):
        z = torch.cat([self.seq(seq_emb), tabular_feats], dim=1)
        return self.head(z).squeeze(-1)     # logits

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-28-ch28.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# شبكة GRU مع mask للقنوات الناقصة (sensor dropouts)
import torch
from torch import nn

class SensorGRU(nn.Module):
    def __init__(self, n_sensors=12, hidden=64, n_out=1):
        super().__init__()
        self.gru = nn.GRU(n_sensors, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_out)

    def forward(self, x, mask=None):
        # x: (B, T, n_sensors)   mask: (B, T) 1 للقناة المتاحة
        lengths = mask.sum(dim=1).clamp(min=1) if mask is not None else None
        out, _ = self.gru(x)
        if lengths is not None:
            idx = (lengths - 1).long()
            pooled = out[torch.arange(out.size(0)), idx]
        else:
            pooled = out[:, -1]
        return self.head(pooled).squeeze(-1)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/pytorch-keras-deep-learning/notebooks/ch-28-ch28.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# بنية RAG مختصرة: استرجاع ثم توليد مع اقتباسات
def answer(question, retriever, generator, k=5, score_threshold=0.35):
    hits = retriever.search(question, k=k)
    if not hits or hits[0]["score"] < score_threshold:
        return {"answer": "لا أعرف من الوثائق المتوفّرة.",
                "citations": [], "confidence": "low"}

    context = "\n\n".join(
        f"[{i+1}] {h['text']} (المصدر: {h['source']})"
        for i, h in enumerate(hits)
    )
    prompt = (
        "أجب عن السؤال اعتماداً على المقاطع التالية فقط. "
        "إذا لم تكفِ المقاطع، أجب: لا أعلم.\n\n"
        f"المقاطع:\n{context}\n\nالسؤال: {question}\nالإجابة:"
    )
    text = generator.generate(prompt)
    return {"answer": text,
            "citations": [h["source"] for h in hits],
            "confidence": "high" if hits[0]["score"] > 0.6 else "medium"}
