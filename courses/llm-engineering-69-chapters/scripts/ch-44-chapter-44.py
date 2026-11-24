#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — النماذج متعددة الوسائط الموحّدة (Unified Multimodal Models)
Large Language Model Engineering — Unified

Book slug : llm-engineering-69-chapters
Chapter   : ch-44-chapter-44 (lesson 44)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-44-chapter-44.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — النماذج متعددة الوسائط الموحّدة (Unified Multimodal Models)
Large Language Model Engineering — Unified

Book slug : llm-engineering-69-chapters
Chapter   : ch-44-chapter-44 (lesson 44)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-44-chapter-44.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — النماذج متعددة الوسائط الموحّدة (Unified Multimodal Models)
Large Language Model Engineering — Unified

Book slug : llm-engineering-69-chapters
Chapter   : ch-44-chapter-44 (lesson 44)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_44.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 44: الفصل 44: النماذج متعددة الوسائط الموحّدة (Unified Multimodal Models)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_multimodal() -> None:
    sequence = [('text', 'hello'), ('image', 'patch_01'), ('image', 'patch_02'), ('audio', 'frame_01')]
    modality_ids = {'text': 0, 'image': 1, 'audio': 2}
    encoded = [(modality_ids[modality], payload) for modality, payload in sequence]
    print('typed sequence:', encoded)
    print('modality counts:', dict(Counter((modality for modality, _ in sequence))))
    assert all((0 <= token_id < len(modality_ids) for token_id, _ in encoded))

if __name__ == "__main__":
    print("Chapter 44 lab: Pack text, image, and audio into typed tokens")
    run_multimodal()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-08-multimodal-models/chapter-44.md
# ────────────────────────────────────────────────────────────────────────
# توضيح مفاهيمي (غير مخصص للتشغيل المباشر) لقناع الانتباه الهجين في Transfusion
def build_hybrid_attention_mask(block_types: list[str], block_lengths: list[int]) -> torch.Tensor:
    """
    block_types: قائمة من "text" أو "image" لكل كتلة في التسلسل
    block_lengths: طول كل كتلة بالرموز/الرقع
    يُرجع قناع انتباه (T, T): سببي بين الكتل، ثنائي الاتجاه داخل كتل الصور
    """
    total_len = sum(block_lengths)
    mask = torch.tril(torch.ones(total_len, total_len))  # سببي كخط أساس

    offset = 0
    for block_type, length in zip(block_types, block_lengths):
        if block_type == "image":
            # افتح الانتباه الثنائي الاتجاه بين رقع نفس كتلة الصورة فقط
            mask[offset:offset+length, offset:offset+length] = 1
        offset += length
    return mask

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-08-multimodal-models/chapter-44.md
# ────────────────────────────────────────────────────────────────────────
"""
MiniUnified: نموذج موحّد مصغّر يتعامل مع النص والصورة كرموز في قاموس واحد.
مستوحى من فلسفة Chameleon (Meta, 2024): Early-Fusion Token-Based Architecture.
"""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

torch.manual_seed(42)
random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"الجهاز المستخدم: {device}")

# ============================================================
# الجزء 1: VQ-VAE — تحويل الصور إلى "كلمات بصرية" منفصلة
# ============================================================

class VectorQuantizer(nn.Module):
    """طبقة التكميم المتجهي: تحوّل متجهات مستمرة إلى أقرب 'كلمة' في قاموس متعلَّم."""

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z: torch.Tensor):
        # z: (B, C, H, W) -> (B, H, W, C) لتسهيل البحث عن أقرب جار لكل موضع مكاني
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flat = z.view(-1, self.embedding_dim)

        # المسافة الإقليدية بين كل متجه وكل "كلمة" في القاموس
        distances = (
            z_flat.pow(2).sum(1, keepdim=True)
            - 2 * z_flat @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(1)
        )
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.codebook(encoding_indices).view(z.shape)

        # خسارتا القاموس والالتزام (انظر الاشتقاق في القسم 🟡 أعلاه)
        codebook_loss = F.mse_loss(quantized, z.detach())
        commitment_loss = F.mse_loss(quantized.detach(), z)
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # مُقدِّر المرور المباشر (Straight-Through Estimator)
        quantized_st = z + (quantized - z).detach()
        quantized_st = quantized_st.permute(0, 3, 1, 2).contiguous()

        encoding_indices = encoding_indices.view(z.shape[0], z.shape[1], z.shape[2])
        return quantized_st, vq_loss, encoding_indices


class VQVAE(nn.Module):
    """VQ-VAE مصغّر: يحوّل صورة MNIST (28x28) إلى خريطة 7x7 من الرموز المنفصلة."""

    def __init__(self, in_channels: int = 1, hidden_dim: int = 64,
                 embedding_dim: int = 64, num_embeddings: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 4, stride=2, padding=1),   # 28 -> 14
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 4, stride=2, padding=1),    # 14 -> 7
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, embedding_dim, 3, stride=1, padding=1),  # يبقى 7x7
        )
        self.vq = VectorQuantizer(num_embeddings, embedding_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, hidden_dim, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1),  # 7 -> 14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, in_channels, 4, stride=2, padding=1),  # 14 -> 28
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        quantized, vq_loss, indices = self.vq(z)
        x_recon = self.decoder(quantized)
        return x_recon, vq_loss, indices

    @torch.no_grad()
    def encode_to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """يحوّل صورة إلى خريطة رموز (B, 7, 7) من مؤشرات القاموس."""
        z = self.encoder(x)
        _, _, indices = self.vq(z)
        return indices

    @torch.no_grad()
    def decode_from_tokens(self, indices: torch.Tensor) -> torch.Tensor:
        """يعيد بناء صورة من خريطة رموز (B, 7, 7)."""
        quantized = self.vq.codebook(indices)          # (B, 7, 7, C)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return self.decoder(quantized)


def train_vqvae(epochs: int = 8) -> VQVAE:
    """يدرّب VQ-VAE على MNIST حتى يتقن ضغط الصور إلى 49 رمزًا منفصلًا."""
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2)

    model = VQVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    for epoch in range(epochs):
        total_recon, total_vq = 0.0, 0.0
        for x, _ in loader:
            x = x.to(device)
            x_recon, vq_loss, _ = model(x)
            recon_loss = F.mse_loss(x_recon, x)
            loss = recon_loss + vq_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_recon += recon_loss.item()
            total_vq += vq_loss.item()

        n = len(loader)
        print(f"[VQ-VAE] حقبة {epoch+1}/{epochs} | إعادة بناء: {total_recon/n:.4f} | VQ: {total_vq/n:.4f}")

    return model


# ============================================================
# الجزء 2: القاموس الموحّد (نص + صورة في مساحة رموز واحدة)
# ============================================================

TEXT_TOKENS = [
    "<PAD>", "<BOS>", "<EOS>", "<SEP>", "<DIGIT>", "<IMG>",
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
]
TEXT_VOCAB_SIZE = len(TEXT_TOKENS)          # = 16
TOKEN2ID = {t: i for i, t in enumerate(TEXT_TOKENS)}
ID2TOKEN = {i: t for i, t in enumerate(TEXT_TOKENS)}

PAD_ID, BOS_ID, EOS_ID, SEP_ID = TOKEN2ID["<PAD>"], TOKEN2ID["<BOS>"], TOKEN2ID["<EOS>"], TOKEN2ID["<SEP>"]
DIGIT_MARK_ID, IMG_MARK_ID = TOKEN2ID["<DIGIT>"], TOKEN2ID["<IMG>"]

DIGIT_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

NUM_IMAGE_CODES = 256         # حجم قاموس VQ-VAE (نفس num_embeddings أعلاه)
IMAGE_TOKEN_OFFSET = TEXT_VOCAB_SIZE          # الرموز البصرية تبدأ بعد مفردات النص
TOTAL_VOCAB_SIZE = TEXT_VOCAB_SIZE + NUM_IMAGE_CODES  # = 272

NUM_IMAGE_TOKENS = 7 * 7      # = 49 رمزًا لكل صورة (خريطة VQ-VAE)
# أطول تسلسل: [BOS] <IMG> 49 رمز [SEP] <DIGIT> word [EOS] = 55 رمزًا
MAX_SEQ_LEN = 1 + 1 + NUM_IMAGE_TOKENS + 1 + 1 + 1 + 1

print(f"حجم القاموس الموحّد: {TOTAL_VOCAB_SIZE} (نص: {TEXT_VOCAB_SIZE} + صورة: {NUM_IMAGE_CODES})")


# ============================================================
# الجزء 3: بناء بيانات ثنائية الاتجاه (نص↔صورة) من نفس العينة
# ============================================================

class UnifiedMultimodalDataset(Dataset):
    """
    لكل صورة MNIST، نبني تسلسلين تدريبيين مختلفي الاتجاه:
      1) نص -> صورة:  [BOS] <DIGIT> word <SEP> [49 رمز صورة] [EOS]
      2) صورة -> نص:  [BOS] <IMG> [49 رمز صورة] <SEP> <DIGIT> word [EOS]
    هذا يُعلّم نموذجًا واحدًا كلا الاتجاهين بنفس الأوزان — جوهر "التوحيد".
    """

    def __init__(self, mnist_dataset, vqvae: VQVAE, max_samples: int | None = None):
        self.sequences: list[torch.Tensor] = []
        vqvae.eval()
        loader = DataLoader(mnist_dataset, batch_size=512, shuffle=False)

        count = 0
        for images, labels in loader:
            images = images.to(device)
            token_maps = vqvae.encode_to_tokens(images)                 # (B, 7, 7)
            token_maps = token_maps.view(token_maps.size(0), -1)        # (B, 49)
            token_maps = (token_maps + IMAGE_TOKEN_OFFSET).cpu()

            for img_tokens, label in zip(token_maps, labels):
                word_id = TOKEN2ID[DIGIT_WORDS[label.item()]]
                img_list = img_tokens.tolist()

                seq_text_to_image = [BOS_ID, DIGIT_MARK_ID, word_id, SEP_ID] + img_list + [EOS_ID]
                seq_image_to_text = [BOS_ID, IMG_MARK_ID] + img_list + [SEP_ID, DIGIT_MARK_ID, word_id, EOS_ID]

                self.sequences.append(torch.tensor(seq_text_to_image, dtype=torch.long))
                self.sequences.append(torch.tensor(seq_image_to_text, dtype=torch.long))

                count += 1
                if max_samples and count >= max_samples:
                    return

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def collate_fn(batch: list[torch.Tensor]) -> torch.Tensor:
    """حشو كل التسلسلات إلى MAX_SEQ_LEN برمز <PAD>."""
    padded = torch.full((len(batch), MAX_SEQ_LEN), PAD_ID, dtype=torch.long)
    for i, seq in enumerate(batch):
        padded[i, :len(seq)] = seq
    return padded


# ============================================================
# الجزء 4: الترانسفورمر الموحّد (Decoder-Only) — قلب النظام
# ============================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # كل منها: (B, n_heads, T, head_dim)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(attn_mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model), nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class UnifiedMultimodalTransformer(nn.Module):
    """
    ترانسفورمر واحد يتعامل مع كل الرموز (نص أو صورة) بنفس الطريقة بالضبط —
    هذا هو جوهر معمارية Early-Fusion (مستوحاة من Chameleon).
    """

    def __init__(self, vocab_size: int, d_model: int = 256, n_heads: int = 8,
                 n_layers: int = 6, d_ff: int = 1024, max_seq_len: int = MAX_SEQ_LEN,
                 dropout: float = 0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_seq_len = max_seq_len

        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("causal_mask", causal_mask.view(1, 1, max_seq_len, max_seq_len))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)

        mask = self.causal_mask[:, :, :T, :T]
        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_f(x)
        return self.head(x)


def train_unified_model(model: UnifiedMultimodalTransformer, dataloader: DataLoader,
                         epochs: int = 15, lr: float = 3e-4) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)
            inputs, targets = batch[:, :-1], batch[:, 1:]

            logits = model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=PAD_ID
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        print(f"[موحّد] حقبة {epoch+1}/{epochs} | الخسارة: {total_loss/len(dataloader):.4f}")


# ============================================================
# الجزء 5: التوليد ثنائي الاتجاه — نفس النموذج، مهمتان مختلفتان
# ============================================================

@torch.no_grad()
def generate_image_from_digit(model: UnifiedMultimodalTransformer, vqvae: VQVAE,
                               digit_word: str) -> torch.Tensor:
    """التوليد النصي -> البصري: 'اكتب لي الرقم سبعة' فتولّد الشبكة صورة."""
    model.eval()
    word_id = TOKEN2ID[digit_word]
    seq = torch.tensor([[BOS_ID, DIGIT_MARK_ID, word_id, SEP_ID]], device=device)

    for _ in range(NUM_IMAGE_TOKENS):
        logits = model(seq)[:, -1, :]
        logits[:, :IMAGE_TOKEN_OFFSET] = float("-inf")   # اقصر الاختيار على رموز الصورة فقط
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        seq = torch.cat([seq, next_token], dim=1)

    image_tokens = seq[0, 4:4 + NUM_IMAGE_TOKENS] - IMAGE_TOKEN_OFFSET
    image_tokens = image_tokens.view(1, 7, 7)
    return vqvae.decode_from_tokens(image_tokens)


@torch.no_grad()
def generate_caption_from_image(model: UnifiedMultimodalTransformer, vqvae: VQVAE,
                                 image: torch.Tensor, max_new_tokens: int = 3) -> str:
    """التوليد البصري -> النصي: صورة رقم مكتوب بخط اليد فيصفها النموذج نصًا."""
    model.eval()
    image = image.to(device).unsqueeze(0)
    img_tokens = vqvae.encode_to_tokens(image).view(1, -1) + IMAGE_TOKEN_OFFSET

    seq = torch.cat([
        torch.tensor([[BOS_ID, IMG_MARK_ID]], device=device),
        img_tokens,
        torch.tensor([[SEP_ID]], device=device),
    ], dim=1)

    generated_ids = []
    for _ in range(max_new_tokens):
        logits = model(seq)[:, -1, :]
        logits[:, IMAGE_TOKEN_OFFSET:] = float("-inf")  # اقصر الاختيار على مفردات النص فقط
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        seq = torch.cat([seq, next_id], dim=1)
        if next_id.item() == EOS_ID:
            break
        generated_ids.append(next_id.item())

    words = [ID2TOKEN[t] for t in generated_ids if t != DIGIT_MARK_ID]
    return " ".join(words)


# ============================================================
# الجزء 6: تشغيل خط الأنابيب الكامل
# ============================================================

if __name__ == "__main__":
    print("\n===== المرحلة 1: تدريب VQ-VAE =====")
    vqvae = train_vqvae(epochs=8)
    for p in vqvae.parameters():
        p.requires_grad_(False)   # نُجمّد VQ-VAE بعد تدريبه — مثل تجميد مُرمِّز الصور في LLaVA

    print("\n===== المرحلة 2: بناء بيانات التدريب الموحّدة =====")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    unified_dataset = UnifiedMultimodalDataset(mnist_train, vqvae, max_samples=6000)
    unified_loader = DataLoader(unified_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    print(f"عدد التسلسلات التدريبية (الاتجاهان معًا): {len(unified_dataset)}")

    print("\n===== المرحلة 3: تدريب الترانسفورمر الموحّد =====")
    unified_model = UnifiedMultimodalTransformer(vocab_size=TOTAL_VOCAB_SIZE).to(device)
    train_unified_model(unified_model, unified_loader, epochs=15)

    print("\n===== المرحلة 4: التوليد ثنائي الاتجاه بنفس الأوزان =====")

    # (أ) نص -> صورة
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, word in enumerate(DIGIT_WORDS):
        img = generate_image_from_digit(unified_model, vqvae, word)
        img = (img.squeeze().cpu() * 0.5 + 0.5).clamp(0, 1)
        ax = axes[i // 5, i % 5]
        ax.imshow(img, cmap="gray")
        ax.set_title(f"'{word}'")
        ax.axis("off")
    plt.suptitle("توليد صور من وصف نصي (نفس أوزان النموذج)")
    plt.tight_layout()
    plt.savefig("unified_text_to_image.png", dpi=120)
    print("تم حفظ: unified_text_to_image.png")

    # (ب) صورة -> نص
    mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    print("\nأمثلة على التوصيف (صورة -> نص):")
    for i in range(5):
        image, true_label = mnist_test[i]
        caption = generate_caption_from_image(unified_model, vqvae, image)
        print(f"  الرقم الحقيقي: {true_label} | وصف النموذج: '{caption}'")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-44-chapter-44.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 44: الفصل 44: النماذج متعددة الوسائط الموحّدة (Unified Multimodal Models)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_multimodal() -> None:
    sequence = [('text', 'hello'), ('image', 'patch_01'), ('image', 'patch_02'), ('audio', 'frame_01')]
    modality_ids = {'text': 0, 'image': 1, 'audio': 2}
    encoded = [(modality_ids[modality], payload) for modality, payload in sequence]
    print('typed sequence:', encoded)
    print('modality counts:', dict(Counter((modality for modality, _ in sequence))))
    assert all((0 <= token_id < len(modality_ids) for token_id, _ in encoded))

if __name__ == "__main__":
    print("Chapter 44 lab: Pack text, image, and audio into typed tokens")
    run_multimodal()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-44-chapter-44.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# توضيح مفاهيمي (غير مخصص للتشغيل المباشر) لقناع الانتباه الهجين في Transfusion
def build_hybrid_attention_mask(block_types: list[str], block_lengths: list[int]) -> torch.Tensor:
    """
    block_types: قائمة من "text" أو "image" لكل كتلة في التسلسل
    block_lengths: طول كل كتلة بالرموز/الرقع
    يُرجع قناع انتباه (T, T): سببي بين الكتل، ثنائي الاتجاه داخل كتل الصور
    """
    total_len = sum(block_lengths)
    mask = torch.tril(torch.ones(total_len, total_len))  # سببي كخط أساس

    offset = 0
    for block_type, length in zip(block_types, block_lengths):
        if block_type == "image":
            # افتح الانتباه الثنائي الاتجاه بين رقع نفس كتلة الصورة فقط
            mask[offset:offset+length, offset:offset+length] = 1
        offset += length
    return mask

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-44-chapter-44.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
MiniUnified: نموذج موحّد مصغّر يتعامل مع النص والصورة كرموز في قاموس واحد.
مستوحى من فلسفة Chameleon (Meta, 2024): Early-Fusion Token-Based Architecture.
"""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

torch.manual_seed(42)
random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"الجهاز المستخدم: {device}")

# ============================================================
# الجزء 1: VQ-VAE — تحويل الصور إلى "كلمات بصرية" منفصلة
# ============================================================

class VectorQuantizer(nn.Module):
    """طبقة التكميم المتجهي: تحوّل متجهات مستمرة إلى أقرب 'كلمة' في قاموس متعلَّم."""

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z: torch.Tensor):
        # z: (B, C, H, W) -> (B, H, W, C) لتسهيل البحث عن أقرب جار لكل موضع مكاني
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flat = z.view(-1, self.embedding_dim)

        # المسافة الإقليدية بين كل متجه وكل "كلمة" في القاموس
        distances = (
            z_flat.pow(2).sum(1, keepdim=True)
            - 2 * z_flat @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(1)
        )
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.codebook(encoding_indices).view(z.shape)

        # خسارتا القاموس والالتزام (انظر الاشتقاق في القسم 🟡 أعلاه)
        codebook_loss = F.mse_loss(quantized, z.detach())
        commitment_loss = F.mse_loss(quantized.detach(), z)
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # مُقدِّر المرور المباشر (Straight-Through Estimator)
        quantized_st = z + (quantized - z).detach()
        quantized_st = quantized_st.permute(0, 3, 1, 2).contiguous()

        encoding_indices = encoding_indices.view(z.shape[0], z.shape[1], z.shape[2])
        return quantized_st, vq_loss, encoding_indices


class VQVAE(nn.Module):
    """VQ-VAE مصغّر: يحوّل صورة MNIST (28x28) إلى خريطة 7x7 من الرموز المنفصلة."""

    def __init__(self, in_channels: int = 1, hidden_dim: int = 64,
                 embedding_dim: int = 64, num_embeddings: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 4, stride=2, padding=1),   # 28 -> 14
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 4, stride=2, padding=1),    # 14 -> 7
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, embedding_dim, 3, stride=1, padding=1),  # يبقى 7x7
        )
        self.vq = VectorQuantizer(num_embeddings, embedding_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, hidden_dim, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1),  # 7 -> 14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, in_channels, 4, stride=2, padding=1),  # 14 -> 28
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        quantized, vq_loss, indices = self.vq(z)
        x_recon = self.decoder(quantized)
        return x_recon, vq_loss, indices

    @torch.no_grad()
    def encode_to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """يحوّل صورة إلى خريطة رموز (B, 7, 7) من مؤشرات القاموس."""
        z = self.encoder(x)
        _, _, indices = self.vq(z)
        return indices

    @torch.no_grad()
    def decode_from_tokens(self, indices: torch.Tensor) -> torch.Tensor:
        """يعيد بناء صورة من خريطة رموز (B, 7, 7)."""
        quantized = self.vq.codebook(indices)          # (B, 7, 7, C)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return self.decoder(quantized)


def train_vqvae(epochs: int = 8) -> VQVAE:
    """يدرّب VQ-VAE على MNIST حتى يتقن ضغط الصور إلى 49 رمزًا منفصلًا."""
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2)

    model = VQVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    for epoch in range(epochs):
        total_recon, total_vq = 0.0, 0.0
        for x, _ in loader:
            x = x.to(device)
            x_recon, vq_loss, _ = model(x)
            recon_loss = F.mse_loss(x_recon, x)
            loss = recon_loss + vq_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_recon += recon_loss.item()
            total_vq += vq_loss.item()

        n = len(loader)
        print(f"[VQ-VAE] حقبة {epoch+1}/{epochs} | إعادة بناء: {total_recon/n:.4f} | VQ: {total_vq/n:.4f}")

    return model


# ============================================================
# الجزء 2: القاموس الموحّد (نص + صورة في مساحة رموز واحدة)
# ============================================================

TEXT_TOKENS = [
    "<PAD>", "<BOS>", "<EOS>", "<SEP>", "<DIGIT>", "<IMG>",
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
]
TEXT_VOCAB_SIZE = len(TEXT_TOKENS)          # = 16
TOKEN2ID = {t: i for i, t in enumerate(TEXT_TOKENS)}
ID2TOKEN = {i: t for i, t in enumerate(TEXT_TOKENS)}

PAD_ID, BOS_ID, EOS_ID, SEP_ID = TOKEN2ID["<PAD>"], TOKEN2ID["<BOS>"], TOKEN2ID["<EOS>"], TOKEN2ID["<SEP>"]
DIGIT_MARK_ID, IMG_MARK_ID = TOKEN2ID["<DIGIT>"], TOKEN2ID["<IMG>"]

DIGIT_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

NUM_IMAGE_CODES = 256         # حجم قاموس VQ-VAE (نفس num_embeddings أعلاه)
IMAGE_TOKEN_OFFSET = TEXT_VOCAB_SIZE          # الرموز البصرية تبدأ بعد مفردات النص
TOTAL_VOCAB_SIZE = TEXT_VOCAB_SIZE + NUM_IMAGE_CODES  # = 272

NUM_IMAGE_TOKENS = 7 * 7      # = 49 رمزًا لكل صورة (خريطة VQ-VAE)
# أطول تسلسل: [BOS] <IMG> 49 رمز [SEP] <DIGIT> word [EOS] = 55 رمزًا
MAX_SEQ_LEN = 1 + 1 + NUM_IMAGE_TOKENS + 1 + 1 + 1 + 1

print(f"حجم القاموس الموحّد: {TOTAL_VOCAB_SIZE} (نص: {TEXT_VOCAB_SIZE} + صورة: {NUM_IMAGE_CODES})")


# ============================================================
# الجزء 3: بناء بيانات ثنائية الاتجاه (نص↔صورة) من نفس العينة
# ============================================================

class UnifiedMultimodalDataset(Dataset):
    """
    لكل صورة MNIST، نبني تسلسلين تدريبيين مختلفي الاتجاه:
      1) نص -> صورة:  [BOS] <DIGIT> word <SEP> [49 رمز صورة] [EOS]
      2) صورة -> نص:  [BOS] <IMG> [49 رمز صورة] <SEP> <DIGIT> word [EOS]
    هذا يُعلّم نموذجًا واحدًا كلا الاتجاهين بنفس الأوزان — جوهر "التوحيد".
    """

    def __init__(self, mnist_dataset, vqvae: VQVAE, max_samples: int | None = None):
        self.sequences: list[torch.Tensor] = []
        vqvae.eval()
        loader = DataLoader(mnist_dataset, batch_size=512, shuffle=False)

        count = 0
        for images, labels in loader:
            images = images.to(device)
            token_maps = vqvae.encode_to_tokens(images)                 # (B, 7, 7)
            token_maps = token_maps.view(token_maps.size(0), -1)        # (B, 49)
            token_maps = (token_maps + IMAGE_TOKEN_OFFSET).cpu()

            for img_tokens, label in zip(token_maps, labels):
                word_id = TOKEN2ID[DIGIT_WORDS[label.item()]]
                img_list = img_tokens.tolist()

                seq_text_to_image = [BOS_ID, DIGIT_MARK_ID, word_id, SEP_ID] + img_list + [EOS_ID]
                seq_image_to_text = [BOS_ID, IMG_MARK_ID] + img_list + [SEP_ID, DIGIT_MARK_ID, word_id, EOS_ID]

                self.sequences.append(torch.tensor(seq_text_to_image, dtype=torch.long))
                self.sequences.append(torch.tensor(seq_image_to_text, dtype=torch.long))

                count += 1
                if max_samples and count >= max_samples:
                    return

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def collate_fn(batch: list[torch.Tensor]) -> torch.Tensor:
    """حشو كل التسلسلات إلى MAX_SEQ_LEN برمز <PAD>."""
    padded = torch.full((len(batch), MAX_SEQ_LEN), PAD_ID, dtype=torch.long)
    for i, seq in enumerate(batch):
        padded[i, :len(seq)] = seq
    return padded


# ============================================================
# الجزء 4: الترانسفورمر الموحّد (Decoder-Only) — قلب النظام
# ============================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # كل منها: (B, n_heads, T, head_dim)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(attn_mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model), nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class UnifiedMultimodalTransformer(nn.Module):
    """
    ترانسفورمر واحد يتعامل مع كل الرموز (نص أو صورة) بنفس الطريقة بالضبط —
    هذا هو جوهر معمارية Early-Fusion (مستوحاة من Chameleon).
    """

    def __init__(self, vocab_size: int, d_model: int = 256, n_heads: int = 8,
                 n_layers: int = 6, d_ff: int = 1024, max_seq_len: int = MAX_SEQ_LEN,
                 dropout: float = 0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_seq_len = max_seq_len

        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("causal_mask", causal_mask.view(1, 1, max_seq_len, max_seq_len))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)

        mask = self.causal_mask[:, :, :T, :T]
        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_f(x)
        return self.head(x)


def train_unified_model(model: UnifiedMultimodalTransformer, dataloader: DataLoader,
                         epochs: int = 15, lr: float = 3e-4) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)
            inputs, targets = batch[:, :-1], batch[:, 1:]

            logits = model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=PAD_ID
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        print(f"[موحّد] حقبة {epoch+1}/{epochs} | الخسارة: {total_loss/len(dataloader):.4f}")


# ============================================================
# الجزء 5: التوليد ثنائي الاتجاه — نفس النموذج، مهمتان مختلفتان
# ============================================================

@torch.no_grad()
def generate_image_from_digit(model: UnifiedMultimodalTransformer, vqvae: VQVAE,
                               digit_word: str) -> torch.Tensor:
    """التوليد النصي -> البصري: 'اكتب لي الرقم سبعة' فتولّد الشبكة صورة."""
    model.eval()
    word_id = TOKEN2ID[digit_word]
    seq = torch.tensor([[BOS_ID, DIGIT_MARK_ID, word_id, SEP_ID]], device=device)

    for _ in range(NUM_IMAGE_TOKENS):
        logits = model(seq)[:, -1, :]
        logits[:, :IMAGE_TOKEN_OFFSET] = float("-inf")   # اقصر الاختيار على رموز الصورة فقط
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        seq = torch.cat([seq, next_token], dim=1)

    image_tokens = seq[0, 4:4 + NUM_IMAGE_TOKENS] - IMAGE_TOKEN_OFFSET
    image_tokens = image_tokens.view(1, 7, 7)
    return vqvae.decode_from_tokens(image_tokens)


@torch.no_grad()
def generate_caption_from_image(model: UnifiedMultimodalTransformer, vqvae: VQVAE,
                                 image: torch.Tensor, max_new_tokens: int = 3) -> str:
    """التوليد البصري -> النصي: صورة رقم مكتوب بخط اليد فيصفها النموذج نصًا."""
    model.eval()
    image = image.to(device).unsqueeze(0)
    img_tokens = vqvae.encode_to_tokens(image).view(1, -1) + IMAGE_TOKEN_OFFSET

    seq = torch.cat([
        torch.tensor([[BOS_ID, IMG_MARK_ID]], device=device),
        img_tokens,
        torch.tensor([[SEP_ID]], device=device),
    ], dim=1)

    generated_ids = []
    for _ in range(max_new_tokens):
        logits = model(seq)[:, -1, :]
        logits[:, IMAGE_TOKEN_OFFSET:] = float("-inf")  # اقصر الاختيار على مفردات النص فقط
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        seq = torch.cat([seq, next_id], dim=1)
        if next_id.item() == EOS_ID:
            break
        generated_ids.append(next_id.item())

    words = [ID2TOKEN[t] for t in generated_ids if t != DIGIT_MARK_ID]
    return " ".join(words)


# ============================================================
# الجزء 6: تشغيل خط الأنابيب الكامل
# ============================================================

if __name__ == "__main__":
    print("\n===== المرحلة 1: تدريب VQ-VAE =====")
    vqvae = train_vqvae(epochs=8)
    for p in vqvae.parameters():
        p.requires_grad_(False)   # نُجمّد VQ-VAE بعد تدريبه — مثل تجميد مُرمِّز الصور في LLaVA

    print("\n===== المرحلة 2: بناء بيانات التدريب الموحّدة =====")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    unified_dataset = UnifiedMultimodalDataset(mnist_train, vqvae, max_samples=6000)
    unified_loader = DataLoader(unified_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    print(f"عدد التسلسلات التدريبية (الاتجاهان معًا): {len(unified_dataset)}")

    print("\n===== المرحلة 3: تدريب الترانسفورمر الموحّد =====")
    unified_model = UnifiedMultimodalTransformer(vocab_size=TOTAL_VOCAB_SIZE).to(device)
    train_unified_model(unified_model, unified_loader, epochs=15)

    print("\n===== المرحلة 4: التوليد ثنائي الاتجاه بنفس الأوزان =====")

    # (أ) نص -> صورة
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, word in enumerate(DIGIT_WORDS):
        img = generate_image_from_digit(unified_model, vqvae, word)
        img = (img.squeeze().cpu() * 0.5 + 0.5).clamp(0, 1)
        ax = axes[i // 5, i % 5]
        ax.imshow(img, cmap="gray")
        ax.set_title(f"'{word}'")
        ax.axis("off")
    plt.suptitle("توليد صور من وصف نصي (نفس أوزان النموذج)")
    plt.tight_layout()
    plt.savefig("unified_text_to_image.png", dpi=120)
    print("تم حفظ: unified_text_to_image.png")

    # (ب) صورة -> نص
    mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    print("\nأمثلة على التوصيف (صورة -> نص):")
    for i in range(5):
        image, true_label = mnist_test[i]
        caption = generate_caption_from_image(unified_model, vqvae, image)
        print(f"  الرقم الحقيقي: {true_label} | وصف النموذج: '{caption}'")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-44-chapter-44.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — النماذج متعددة الوسائط الموحّدة (Unified Multimodal Models)
Large Language Model Engineering — Unified

Book slug : llm-engineering-69-chapters
Chapter   : ch-44-chapter-44 (lesson 44)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_44.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 44: الفصل 44: النماذج متعددة الوسائط الموحّدة (Unified Multimodal Models)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_multimodal() -> None:
    sequence = [('text', 'hello'), ('image', 'patch_01'), ('image', 'patch_02'), ('audio', 'frame_01')]
    modality_ids = {'text': 0, 'image': 1, 'audio': 2}
    encoded = [(modality_ids[modality], payload) for modality, payload in sequence]
    print('typed sequence:', encoded)
    print('modality counts:', dict(Counter((modality for modality, _ in sequence))))
    assert all((0 <= token_id < len(modality_ids) for token_id, _ in encoded))

if __name__ == "__main__":
    print("Chapter 44 lab: Pack text, image, and audio into typed tokens")
    run_multimodal()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-08-multimodal-models/chapter-44.md
# ────────────────────────────────────────────────────────────────────────
# توضيح مفاهيمي (غير مخصص للتشغيل المباشر) لقناع الانتباه الهجين في Transfusion
def build_hybrid_attention_mask(block_types: list[str], block_lengths: list[int]) -> torch.Tensor:
    """
    block_types: قائمة من "text" أو "image" لكل كتلة في التسلسل
    block_lengths: طول كل كتلة بالرموز/الرقع
    يُرجع قناع انتباه (T, T): سببي بين الكتل، ثنائي الاتجاه داخل كتل الصور
    """
    total_len = sum(block_lengths)
    mask = torch.tril(torch.ones(total_len, total_len))  # سببي كخط أساس

    offset = 0
    for block_type, length in zip(block_types, block_lengths):
        if block_type == "image":
            # افتح الانتباه الثنائي الاتجاه بين رقع نفس كتلة الصورة فقط
            mask[offset:offset+length, offset:offset+length] = 1
        offset += length
    return mask

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-08-multimodal-models/chapter-44.md
# ────────────────────────────────────────────────────────────────────────
"""
MiniUnified: نموذج موحّد مصغّر يتعامل مع النص والصورة كرموز في قاموس واحد.
مستوحى من فلسفة Chameleon (Meta, 2024): Early-Fusion Token-Based Architecture.
"""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

torch.manual_seed(42)
random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"الجهاز المستخدم: {device}")

# ============================================================
# الجزء 1: VQ-VAE — تحويل الصور إلى "كلمات بصرية" منفصلة
# ============================================================

class VectorQuantizer(nn.Module):
    """طبقة التكميم المتجهي: تحوّل متجهات مستمرة إلى أقرب 'كلمة' في قاموس متعلَّم."""

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z: torch.Tensor):
        # z: (B, C, H, W) -> (B, H, W, C) لتسهيل البحث عن أقرب جار لكل موضع مكاني
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flat = z.view(-1, self.embedding_dim)

        # المسافة الإقليدية بين كل متجه وكل "كلمة" في القاموس
        distances = (
            z_flat.pow(2).sum(1, keepdim=True)
            - 2 * z_flat @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(1)
        )
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.codebook(encoding_indices).view(z.shape)

        # خسارتا القاموس والالتزام (انظر الاشتقاق في القسم 🟡 أعلاه)
        codebook_loss = F.mse_loss(quantized, z.detach())
        commitment_loss = F.mse_loss(quantized.detach(), z)
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # مُقدِّر المرور المباشر (Straight-Through Estimator)
        quantized_st = z + (quantized - z).detach()
        quantized_st = quantized_st.permute(0, 3, 1, 2).contiguous()

        encoding_indices = encoding_indices.view(z.shape[0], z.shape[1], z.shape[2])
        return quantized_st, vq_loss, encoding_indices


class VQVAE(nn.Module):
    """VQ-VAE مصغّر: يحوّل صورة MNIST (28x28) إلى خريطة 7x7 من الرموز المنفصلة."""

    def __init__(self, in_channels: int = 1, hidden_dim: int = 64,
                 embedding_dim: int = 64, num_embeddings: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 4, stride=2, padding=1),   # 28 -> 14
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 4, stride=2, padding=1),    # 14 -> 7
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, embedding_dim, 3, stride=1, padding=1),  # يبقى 7x7
        )
        self.vq = VectorQuantizer(num_embeddings, embedding_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, hidden_dim, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1),  # 7 -> 14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, in_channels, 4, stride=2, padding=1),  # 14 -> 28
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        quantized, vq_loss, indices = self.vq(z)
        x_recon = self.decoder(quantized)
        return x_recon, vq_loss, indices

    @torch.no_grad()
    def encode_to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """يحوّل صورة إلى خريطة رموز (B, 7, 7) من مؤشرات القاموس."""
        z = self.encoder(x)
        _, _, indices = self.vq(z)
        return indices

    @torch.no_grad()
    def decode_from_tokens(self, indices: torch.Tensor) -> torch.Tensor:
        """يعيد بناء صورة من خريطة رموز (B, 7, 7)."""
        quantized = self.vq.codebook(indices)          # (B, 7, 7, C)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return self.decoder(quantized)


def train_vqvae(epochs: int = 8) -> VQVAE:
    """يدرّب VQ-VAE على MNIST حتى يتقن ضغط الصور إلى 49 رمزًا منفصلًا."""
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2)

    model = VQVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    for epoch in range(epochs):
        total_recon, total_vq = 0.0, 0.0
        for x, _ in loader:
            x = x.to(device)
            x_recon, vq_loss, _ = model(x)
            recon_loss = F.mse_loss(x_recon, x)
            loss = recon_loss + vq_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_recon += recon_loss.item()
            total_vq += vq_loss.item()

        n = len(loader)
        print(f"[VQ-VAE] حقبة {epoch+1}/{epochs} | إعادة بناء: {total_recon/n:.4f} | VQ: {total_vq/n:.4f}")

    return model


# ============================================================
# الجزء 2: القاموس الموحّد (نص + صورة في مساحة رموز واحدة)
# ============================================================

TEXT_TOKENS = [
    "<PAD>", "<BOS>", "<EOS>", "<SEP>", "<DIGIT>", "<IMG>",
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
]
TEXT_VOCAB_SIZE = len(TEXT_TOKENS)          # = 16
TOKEN2ID = {t: i for i, t in enumerate(TEXT_TOKENS)}
ID2TOKEN = {i: t for i, t in enumerate(TEXT_TOKENS)}

PAD_ID, BOS_ID, EOS_ID, SEP_ID = TOKEN2ID["<PAD>"], TOKEN2ID["<BOS>"], TOKEN2ID["<EOS>"], TOKEN2ID["<SEP>"]
DIGIT_MARK_ID, IMG_MARK_ID = TOKEN2ID["<DIGIT>"], TOKEN2ID["<IMG>"]

DIGIT_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

NUM_IMAGE_CODES = 256         # حجم قاموس VQ-VAE (نفس num_embeddings أعلاه)
IMAGE_TOKEN_OFFSET = TEXT_VOCAB_SIZE          # الرموز البصرية تبدأ بعد مفردات النص
TOTAL_VOCAB_SIZE = TEXT_VOCAB_SIZE + NUM_IMAGE_CODES  # = 272

NUM_IMAGE_TOKENS = 7 * 7      # = 49 رمزًا لكل صورة (خريطة VQ-VAE)
# أطول تسلسل: [BOS] <IMG> 49 رمز [SEP] <DIGIT> word [EOS] = 55 رمزًا
MAX_SEQ_LEN = 1 + 1 + NUM_IMAGE_TOKENS + 1 + 1 + 1 + 1

print(f"حجم القاموس الموحّد: {TOTAL_VOCAB_SIZE} (نص: {TEXT_VOCAB_SIZE} + صورة: {NUM_IMAGE_CODES})")


# ============================================================
# الجزء 3: بناء بيانات ثنائية الاتجاه (نص↔صورة) من نفس العينة
# ============================================================

class UnifiedMultimodalDataset(Dataset):
    """
    لكل صورة MNIST، نبني تسلسلين تدريبيين مختلفي الاتجاه:
      1) نص -> صورة:  [BOS] <DIGIT> word <SEP> [49 رمز صورة] [EOS]
      2) صورة -> نص:  [BOS] <IMG> [49 رمز صورة] <SEP> <DIGIT> word [EOS]
    هذا يُعلّم نموذجًا واحدًا كلا الاتجاهين بنفس الأوزان — جوهر "التوحيد".
    """

    def __init__(self, mnist_dataset, vqvae: VQVAE, max_samples: int | None = None):
        self.sequences: list[torch.Tensor] = []
        vqvae.eval()
        loader = DataLoader(mnist_dataset, batch_size=512, shuffle=False)

        count = 0
        for images, labels in loader:
            images = images.to(device)
            token_maps = vqvae.encode_to_tokens(images)                 # (B, 7, 7)
            token_maps = token_maps.view(token_maps.size(0), -1)        # (B, 49)
            token_maps = (token_maps + IMAGE_TOKEN_OFFSET).cpu()

            for img_tokens, label in zip(token_maps, labels):
                word_id = TOKEN2ID[DIGIT_WORDS[label.item()]]
                img_list = img_tokens.tolist()

                seq_text_to_image = [BOS_ID, DIGIT_MARK_ID, word_id, SEP_ID] + img_list + [EOS_ID]
                seq_image_to_text = [BOS_ID, IMG_MARK_ID] + img_list + [SEP_ID, DIGIT_MARK_ID, word_id, EOS_ID]

                self.sequences.append(torch.tensor(seq_text_to_image, dtype=torch.long))
                self.sequences.append(torch.tensor(seq_image_to_text, dtype=torch.long))

                count += 1
                if max_samples and count >= max_samples:
                    return

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def collate_fn(batch: list[torch.Tensor]) -> torch.Tensor:
    """حشو كل التسلسلات إلى MAX_SEQ_LEN برمز <PAD>."""
    padded = torch.full((len(batch), MAX_SEQ_LEN), PAD_ID, dtype=torch.long)
    for i, seq in enumerate(batch):
        padded[i, :len(seq)] = seq
    return padded


# ============================================================
# الجزء 4: الترانسفورمر الموحّد (Decoder-Only) — قلب النظام
# ============================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # كل منها: (B, n_heads, T, head_dim)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(attn_mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model), nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class UnifiedMultimodalTransformer(nn.Module):
    """
    ترانسفورمر واحد يتعامل مع كل الرموز (نص أو صورة) بنفس الطريقة بالضبط —
    هذا هو جوهر معمارية Early-Fusion (مستوحاة من Chameleon).
    """

    def __init__(self, vocab_size: int, d_model: int = 256, n_heads: int = 8,
                 n_layers: int = 6, d_ff: int = 1024, max_seq_len: int = MAX_SEQ_LEN,
                 dropout: float = 0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_seq_len = max_seq_len

        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("causal_mask", causal_mask.view(1, 1, max_seq_len, max_seq_len))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)

        mask = self.causal_mask[:, :, :T, :T]
        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_f(x)
        return self.head(x)


def train_unified_model(model: UnifiedMultimodalTransformer, dataloader: DataLoader,
                         epochs: int = 15, lr: float = 3e-4) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)
            inputs, targets = batch[:, :-1], batch[:, 1:]

            logits = model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=PAD_ID
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        print(f"[موحّد] حقبة {epoch+1}/{epochs} | الخسارة: {total_loss/len(dataloader):.4f}")


# ============================================================
# الجزء 5: التوليد ثنائي الاتجاه — نفس النموذج، مهمتان مختلفتان
# ============================================================

@torch.no_grad()
def generate_image_from_digit(model: UnifiedMultimodalTransformer, vqvae: VQVAE,
                               digit_word: str) -> torch.Tensor:
    """التوليد النصي -> البصري: 'اكتب لي الرقم سبعة' فتولّد الشبكة صورة."""
    model.eval()
    word_id = TOKEN2ID[digit_word]
    seq = torch.tensor([[BOS_ID, DIGIT_MARK_ID, word_id, SEP_ID]], device=device)

    for _ in range(NUM_IMAGE_TOKENS):
        logits = model(seq)[:, -1, :]
        logits[:, :IMAGE_TOKEN_OFFSET] = float("-inf")   # اقصر الاختيار على رموز الصورة فقط
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        seq = torch.cat([seq, next_token], dim=1)

    image_tokens = seq[0, 4:4 + NUM_IMAGE_TOKENS] - IMAGE_TOKEN_OFFSET
    image_tokens = image_tokens.view(1, 7, 7)
    return vqvae.decode_from_tokens(image_tokens)


@torch.no_grad()
def generate_caption_from_image(model: UnifiedMultimodalTransformer, vqvae: VQVAE,
                                 image: torch.Tensor, max_new_tokens: int = 3) -> str:
    """التوليد البصري -> النصي: صورة رقم مكتوب بخط اليد فيصفها النموذج نصًا."""
    model.eval()
    image = image.to(device).unsqueeze(0)
    img_tokens = vqvae.encode_to_tokens(image).view(1, -1) + IMAGE_TOKEN_OFFSET

    seq = torch.cat([
        torch.tensor([[BOS_ID, IMG_MARK_ID]], device=device),
        img_tokens,
        torch.tensor([[SEP_ID]], device=device),
    ], dim=1)

    generated_ids = []
    for _ in range(max_new_tokens):
        logits = model(seq)[:, -1, :]
        logits[:, IMAGE_TOKEN_OFFSET:] = float("-inf")  # اقصر الاختيار على مفردات النص فقط
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        seq = torch.cat([seq, next_id], dim=1)
        if next_id.item() == EOS_ID:
            break
        generated_ids.append(next_id.item())

    words = [ID2TOKEN[t] for t in generated_ids if t != DIGIT_MARK_ID]
    return " ".join(words)


# ============================================================
# الجزء 6: تشغيل خط الأنابيب الكامل
# ============================================================

if __name__ == "__main__":
    print("\n===== المرحلة 1: تدريب VQ-VAE =====")
    vqvae = train_vqvae(epochs=8)
    for p in vqvae.parameters():
        p.requires_grad_(False)   # نُجمّد VQ-VAE بعد تدريبه — مثل تجميد مُرمِّز الصور في LLaVA

    print("\n===== المرحلة 2: بناء بيانات التدريب الموحّدة =====")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    unified_dataset = UnifiedMultimodalDataset(mnist_train, vqvae, max_samples=6000)
    unified_loader = DataLoader(unified_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    print(f"عدد التسلسلات التدريبية (الاتجاهان معًا): {len(unified_dataset)}")

    print("\n===== المرحلة 3: تدريب الترانسفورمر الموحّد =====")
    unified_model = UnifiedMultimodalTransformer(vocab_size=TOTAL_VOCAB_SIZE).to(device)
    train_unified_model(unified_model, unified_loader, epochs=15)

    print("\n===== المرحلة 4: التوليد ثنائي الاتجاه بنفس الأوزان =====")

    # (أ) نص -> صورة
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, word in enumerate(DIGIT_WORDS):
        img = generate_image_from_digit(unified_model, vqvae, word)
        img = (img.squeeze().cpu() * 0.5 + 0.5).clamp(0, 1)
        ax = axes[i // 5, i % 5]
        ax.imshow(img, cmap="gray")
        ax.set_title(f"'{word}'")
        ax.axis("off")
    plt.suptitle("توليد صور من وصف نصي (نفس أوزان النموذج)")
    plt.tight_layout()
    plt.savefig("unified_text_to_image.png", dpi=120)
    print("تم حفظ: unified_text_to_image.png")

    # (ب) صورة -> نص
    mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    print("\nأمثلة على التوصيف (صورة -> نص):")
    for i in range(5):
        image, true_label = mnist_test[i]
        caption = generate_caption_from_image(unified_model, vqvae, image)
        print(f"  الرقم الحقيقي: {true_label} | وصف النموذج: '{caption}'")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-44-chapter-44.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 44: الفصل 44: النماذج متعددة الوسائط الموحّدة (Unified Multimodal Models)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_multimodal() -> None:
    sequence = [('text', 'hello'), ('image', 'patch_01'), ('image', 'patch_02'), ('audio', 'frame_01')]
    modality_ids = {'text': 0, 'image': 1, 'audio': 2}
    encoded = [(modality_ids[modality], payload) for modality, payload in sequence]
    print('typed sequence:', encoded)
    print('modality counts:', dict(Counter((modality for modality, _ in sequence))))
    assert all((0 <= token_id < len(modality_ids) for token_id, _ in encoded))

if __name__ == "__main__":
    print("Chapter 44 lab: Pack text, image, and audio into typed tokens")
    run_multimodal()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-44-chapter-44.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# توضيح مفاهيمي (غير مخصص للتشغيل المباشر) لقناع الانتباه الهجين في Transfusion
def build_hybrid_attention_mask(block_types: list[str], block_lengths: list[int]) -> torch.Tensor:
    """
    block_types: قائمة من "text" أو "image" لكل كتلة في التسلسل
    block_lengths: طول كل كتلة بالرموز/الرقع
    يُرجع قناع انتباه (T, T): سببي بين الكتل، ثنائي الاتجاه داخل كتل الصور
    """
    total_len = sum(block_lengths)
    mask = torch.tril(torch.ones(total_len, total_len))  # سببي كخط أساس

    offset = 0
    for block_type, length in zip(block_types, block_lengths):
        if block_type == "image":
            # افتح الانتباه الثنائي الاتجاه بين رقع نفس كتلة الصورة فقط
            mask[offset:offset+length, offset:offset+length] = 1
        offset += length
    return mask

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-44-chapter-44.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
"""
MiniUnified: نموذج موحّد مصغّر يتعامل مع النص والصورة كرموز في قاموس واحد.
مستوحى من فلسفة Chameleon (Meta, 2024): Early-Fusion Token-Based Architecture.
"""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

torch.manual_seed(42)
random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"الجهاز المستخدم: {device}")

# ============================================================
# الجزء 1: VQ-VAE — تحويل الصور إلى "كلمات بصرية" منفصلة
# ============================================================

class VectorQuantizer(nn.Module):
    """طبقة التكميم المتجهي: تحوّل متجهات مستمرة إلى أقرب 'كلمة' في قاموس متعلَّم."""

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z: torch.Tensor):
        # z: (B, C, H, W) -> (B, H, W, C) لتسهيل البحث عن أقرب جار لكل موضع مكاني
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flat = z.view(-1, self.embedding_dim)

        # المسافة الإقليدية بين كل متجه وكل "كلمة" في القاموس
        distances = (
            z_flat.pow(2).sum(1, keepdim=True)
            - 2 * z_flat @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(1)
        )
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.codebook(encoding_indices).view(z.shape)

        # خسارتا القاموس والالتزام (انظر الاشتقاق في القسم 🟡 أعلاه)
        codebook_loss = F.mse_loss(quantized, z.detach())
        commitment_loss = F.mse_loss(quantized.detach(), z)
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # مُقدِّر المرور المباشر (Straight-Through Estimator)
        quantized_st = z + (quantized - z).detach()
        quantized_st = quantized_st.permute(0, 3, 1, 2).contiguous()

        encoding_indices = encoding_indices.view(z.shape[0], z.shape[1], z.shape[2])
        return quantized_st, vq_loss, encoding_indices


class VQVAE(nn.Module):
    """VQ-VAE مصغّر: يحوّل صورة MNIST (28x28) إلى خريطة 7x7 من الرموز المنفصلة."""

    def __init__(self, in_channels: int = 1, hidden_dim: int = 64,
                 embedding_dim: int = 64, num_embeddings: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 4, stride=2, padding=1),   # 28 -> 14
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 4, stride=2, padding=1),    # 14 -> 7
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, embedding_dim, 3, stride=1, padding=1),  # يبقى 7x7
        )
        self.vq = VectorQuantizer(num_embeddings, embedding_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, hidden_dim, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1),  # 7 -> 14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, in_channels, 4, stride=2, padding=1),  # 14 -> 28
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        quantized, vq_loss, indices = self.vq(z)
        x_recon = self.decoder(quantized)
        return x_recon, vq_loss, indices

    @torch.no_grad()
    def encode_to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """يحوّل صورة إلى خريطة رموز (B, 7, 7) من مؤشرات القاموس."""
        z = self.encoder(x)
        _, _, indices = self.vq(z)
        return indices

    @torch.no_grad()
    def decode_from_tokens(self, indices: torch.Tensor) -> torch.Tensor:
        """يعيد بناء صورة من خريطة رموز (B, 7, 7)."""
        quantized = self.vq.codebook(indices)          # (B, 7, 7, C)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return self.decoder(quantized)


def train_vqvae(epochs: int = 8) -> VQVAE:
    """يدرّب VQ-VAE على MNIST حتى يتقن ضغط الصور إلى 49 رمزًا منفصلًا."""
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2)

    model = VQVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    for epoch in range(epochs):
        total_recon, total_vq = 0.0, 0.0
        for x, _ in loader:
            x = x.to(device)
            x_recon, vq_loss, _ = model(x)
            recon_loss = F.mse_loss(x_recon, x)
            loss = recon_loss + vq_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_recon += recon_loss.item()
            total_vq += vq_loss.item()

        n = len(loader)
        print(f"[VQ-VAE] حقبة {epoch+1}/{epochs} | إعادة بناء: {total_recon/n:.4f} | VQ: {total_vq/n:.4f}")

    return model


# ============================================================
# الجزء 2: القاموس الموحّد (نص + صورة في مساحة رموز واحدة)
# ============================================================

TEXT_TOKENS = [
    "<PAD>", "<BOS>", "<EOS>", "<SEP>", "<DIGIT>", "<IMG>",
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
]
TEXT_VOCAB_SIZE = len(TEXT_TOKENS)          # = 16
TOKEN2ID = {t: i for i, t in enumerate(TEXT_TOKENS)}
ID2TOKEN = {i: t for i, t in enumerate(TEXT_TOKENS)}

PAD_ID, BOS_ID, EOS_ID, SEP_ID = TOKEN2ID["<PAD>"], TOKEN2ID["<BOS>"], TOKEN2ID["<EOS>"], TOKEN2ID["<SEP>"]
DIGIT_MARK_ID, IMG_MARK_ID = TOKEN2ID["<DIGIT>"], TOKEN2ID["<IMG>"]

DIGIT_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

NUM_IMAGE_CODES = 256         # حجم قاموس VQ-VAE (نفس num_embeddings أعلاه)
IMAGE_TOKEN_OFFSET = TEXT_VOCAB_SIZE          # الرموز البصرية تبدأ بعد مفردات النص
TOTAL_VOCAB_SIZE = TEXT_VOCAB_SIZE + NUM_IMAGE_CODES  # = 272

NUM_IMAGE_TOKENS = 7 * 7      # = 49 رمزًا لكل صورة (خريطة VQ-VAE)
# أطول تسلسل: [BOS] <IMG> 49 رمز [SEP] <DIGIT> word [EOS] = 55 رمزًا
MAX_SEQ_LEN = 1 + 1 + NUM_IMAGE_TOKENS + 1 + 1 + 1 + 1

print(f"حجم القاموس الموحّد: {TOTAL_VOCAB_SIZE} (نص: {TEXT_VOCAB_SIZE} + صورة: {NUM_IMAGE_CODES})")


# ============================================================
# الجزء 3: بناء بيانات ثنائية الاتجاه (نص↔صورة) من نفس العينة
# ============================================================

class UnifiedMultimodalDataset(Dataset):
    """
    لكل صورة MNIST، نبني تسلسلين تدريبيين مختلفي الاتجاه:
      1) نص -> صورة:  [BOS] <DIGIT> word <SEP> [49 رمز صورة] [EOS]
      2) صورة -> نص:  [BOS] <IMG> [49 رمز صورة] <SEP> <DIGIT> word [EOS]
    هذا يُعلّم نموذجًا واحدًا كلا الاتجاهين بنفس الأوزان — جوهر "التوحيد".
    """

    def __init__(self, mnist_dataset, vqvae: VQVAE, max_samples: int | None = None):
        self.sequences: list[torch.Tensor] = []
        vqvae.eval()
        loader = DataLoader(mnist_dataset, batch_size=512, shuffle=False)

        count = 0
        for images, labels in loader:
            images = images.to(device)
            token_maps = vqvae.encode_to_tokens(images)                 # (B, 7, 7)
            token_maps = token_maps.view(token_maps.size(0), -1)        # (B, 49)
            token_maps = (token_maps + IMAGE_TOKEN_OFFSET).cpu()

            for img_tokens, label in zip(token_maps, labels):
                word_id = TOKEN2ID[DIGIT_WORDS[label.item()]]
                img_list = img_tokens.tolist()

                seq_text_to_image = [BOS_ID, DIGIT_MARK_ID, word_id, SEP_ID] + img_list + [EOS_ID]
                seq_image_to_text = [BOS_ID, IMG_MARK_ID] + img_list + [SEP_ID, DIGIT_MARK_ID, word_id, EOS_ID]

                self.sequences.append(torch.tensor(seq_text_to_image, dtype=torch.long))
                self.sequences.append(torch.tensor(seq_image_to_text, dtype=torch.long))

                count += 1
                if max_samples and count >= max_samples:
                    return

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def collate_fn(batch: list[torch.Tensor]) -> torch.Tensor:
    """حشو كل التسلسلات إلى MAX_SEQ_LEN برمز <PAD>."""
    padded = torch.full((len(batch), MAX_SEQ_LEN), PAD_ID, dtype=torch.long)
    for i, seq in enumerate(batch):
        padded[i, :len(seq)] = seq
    return padded


# ============================================================
# الجزء 4: الترانسفورمر الموحّد (Decoder-Only) — قلب النظام
# ============================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # كل منها: (B, n_heads, T, head_dim)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(attn_mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model), nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class UnifiedMultimodalTransformer(nn.Module):
    """
    ترانسفورمر واحد يتعامل مع كل الرموز (نص أو صورة) بنفس الطريقة بالضبط —
    هذا هو جوهر معمارية Early-Fusion (مستوحاة من Chameleon).
    """

    def __init__(self, vocab_size: int, d_model: int = 256, n_heads: int = 8,
                 n_layers: int = 6, d_ff: int = 1024, max_seq_len: int = MAX_SEQ_LEN,
                 dropout: float = 0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_seq_len = max_seq_len

        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("causal_mask", causal_mask.view(1, 1, max_seq_len, max_seq_len))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)

        mask = self.causal_mask[:, :, :T, :T]
        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_f(x)
        return self.head(x)


def train_unified_model(model: UnifiedMultimodalTransformer, dataloader: DataLoader,
                         epochs: int = 15, lr: float = 3e-4) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)
            inputs, targets = batch[:, :-1], batch[:, 1:]

            logits = model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=PAD_ID
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        print(f"[موحّد] حقبة {epoch+1}/{epochs} | الخسارة: {total_loss/len(dataloader):.4f}")


# ============================================================
# الجزء 5: التوليد ثنائي الاتجاه — نفس النموذج، مهمتان مختلفتان
# ============================================================

@torch.no_grad()
def generate_image_from_digit(model: UnifiedMultimodalTransformer, vqvae: VQVAE,
                               digit_word: str) -> torch.Tensor:
    """التوليد النصي -> البصري: 'اكتب لي الرقم سبعة' فتولّد الشبكة صورة."""
    model.eval()
    word_id = TOKEN2ID[digit_word]
    seq = torch.tensor([[BOS_ID, DIGIT_MARK_ID, word_id, SEP_ID]], device=device)

    for _ in range(NUM_IMAGE_TOKENS):
        logits = model(seq)[:, -1, :]
        logits[:, :IMAGE_TOKEN_OFFSET] = float("-inf")   # اقصر الاختيار على رموز الصورة فقط
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        seq = torch.cat([seq, next_token], dim=1)

    image_tokens = seq[0, 4:4 + NUM_IMAGE_TOKENS] - IMAGE_TOKEN_OFFSET
    image_tokens = image_tokens.view(1, 7, 7)
    return vqvae.decode_from_tokens(image_tokens)


@torch.no_grad()
def generate_caption_from_image(model: UnifiedMultimodalTransformer, vqvae: VQVAE,
                                 image: torch.Tensor, max_new_tokens: int = 3) -> str:
    """التوليد البصري -> النصي: صورة رقم مكتوب بخط اليد فيصفها النموذج نصًا."""
    model.eval()
    image = image.to(device).unsqueeze(0)
    img_tokens = vqvae.encode_to_tokens(image).view(1, -1) + IMAGE_TOKEN_OFFSET

    seq = torch.cat([
        torch.tensor([[BOS_ID, IMG_MARK_ID]], device=device),
        img_tokens,
        torch.tensor([[SEP_ID]], device=device),
    ], dim=1)

    generated_ids = []
    for _ in range(max_new_tokens):
        logits = model(seq)[:, -1, :]
        logits[:, IMAGE_TOKEN_OFFSET:] = float("-inf")  # اقصر الاختيار على مفردات النص فقط
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        seq = torch.cat([seq, next_id], dim=1)
        if next_id.item() == EOS_ID:
            break
        generated_ids.append(next_id.item())

    words = [ID2TOKEN[t] for t in generated_ids if t != DIGIT_MARK_ID]
    return " ".join(words)


# ============================================================
# الجزء 6: تشغيل خط الأنابيب الكامل
# ============================================================

if __name__ == "__main__":
    print("\n===== المرحلة 1: تدريب VQ-VAE =====")
    vqvae = train_vqvae(epochs=8)
    for p in vqvae.parameters():
        p.requires_grad_(False)   # نُجمّد VQ-VAE بعد تدريبه — مثل تجميد مُرمِّز الصور في LLaVA

    print("\n===== المرحلة 2: بناء بيانات التدريب الموحّدة =====")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    unified_dataset = UnifiedMultimodalDataset(mnist_train, vqvae, max_samples=6000)
    unified_loader = DataLoader(unified_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    print(f"عدد التسلسلات التدريبية (الاتجاهان معًا): {len(unified_dataset)}")

    print("\n===== المرحلة 3: تدريب الترانسفورمر الموحّد =====")
    unified_model = UnifiedMultimodalTransformer(vocab_size=TOTAL_VOCAB_SIZE).to(device)
    train_unified_model(unified_model, unified_loader, epochs=15)

    print("\n===== المرحلة 4: التوليد ثنائي الاتجاه بنفس الأوزان =====")

    # (أ) نص -> صورة
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, word in enumerate(DIGIT_WORDS):
        img = generate_image_from_digit(unified_model, vqvae, word)
        img = (img.squeeze().cpu() * 0.5 + 0.5).clamp(0, 1)
        ax = axes[i // 5, i % 5]
        ax.imshow(img, cmap="gray")
        ax.set_title(f"'{word}'")
        ax.axis("off")
    plt.suptitle("توليد صور من وصف نصي (نفس أوزان النموذج)")
    plt.tight_layout()
    plt.savefig("unified_text_to_image.png", dpi=120)
    print("تم حفظ: unified_text_to_image.png")

    # (ب) صورة -> نص
    mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    print("\nأمثلة على التوصيف (صورة -> نص):")
    for i in range(5):
        image, true_label = mnist_test[i]
        caption = generate_caption_from_image(unified_model, vqvae, image)
        print(f"  الرقم الحقيقي: {true_label} | وصف النموذج: '{caption}'")
